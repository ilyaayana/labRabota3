#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QPixmap>
#include <QFileDialog>
#include <QPixmap>
#include <QScrollArea>
#include <QtMath>

using namespace std;
using namespace cv;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

     filename = QFileDialog::getOpenFileName(this).toStdString();
     image = imread(filename);

    setUpImagesAndHistograms();

    ui->scrollArea->resize(width(),height()*0.9);

}
Mat MainWindow::equalImageHist(Mat image){
    Mat hist_equalized_image;

    cvtColor(image, hist_equalized_image, COLOR_BGR2YCrCb);

    vector<Mat> vec_channels;
    split(hist_equalized_image, vec_channels);

    equalizeHist(vec_channels[0], vec_channels[0]);

    merge(vec_channels, hist_equalized_image);

    cvtColor(hist_equalized_image, hist_equalized_image, COLOR_YCrCb2BGR);

    return hist_equalized_image;
}
Mat MainWindow::equalImageHistRGB(Mat image){

    Mat modified_image;
    image.copyTo(modified_image);

    vector<Mat> vec_channels;
    split(modified_image, vec_channels);

    equalizeHist(vec_channels[0], vec_channels[0]);

    merge(vec_channels, modified_image);

    return modified_image;
}

Mat MainWindow::buildHistogram(Mat src){

    vector<Mat> bgr_planes;
    split( src, bgr_planes );
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;
    Mat b_hist, g_hist, r_hist;
    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate );
    calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate );
    calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, histRange, uniform, accumulate );
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );
    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ),
              Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
              Scalar( 255, 0, 0), 2, 8, 0  );
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ),
              Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
              Scalar( 0, 255, 0), 2, 8, 0  );
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ),
              Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
              Scalar( 0, 0, 255), 2, 8, 0  );
    }
    return histImage;
}

Mat MainWindow::linearContrast(Mat image){

    Mat modified_image;
    image.copyTo(modified_image);

    cvtColor(modified_image, modified_image, COLOR_BGR2YCrCb);

    vector<Mat> vec_channels;
    split(modified_image, vec_channels);

        double min,max;
        minMaxLoc(modified_image,&min,&max);
        for(int j=0;j<vec_channels[0].rows;j++)
        {
              for (int i=0;i<vec_channels[0].cols;i++)
              {
                  int x = vec_channels[0].at<uchar>(j,i);
                  vec_channels[0].at<uchar>(j,i) = (x-min)/(max-min)*255;
              }
        }
    merge(vec_channels, modified_image);

    cvtColor(modified_image, modified_image, COLOR_YCrCb2BGR);

    return modified_image;
}
Mat MainWindow::add(int value){

    Mat modified_image;
    image.copyTo(modified_image);

    cvtColor(modified_image, modified_image, COLOR_BGR2YCrCb);

    vector<Mat> vec_channels;
    split(modified_image, vec_channels);

        for(int j=0;j<vec_channels[0].rows;j++)
        {
              for (int i=0;i<vec_channels[0].cols;i++)
              {
                  vec_channels[0].at<uchar>(j,i) += value;
              }
        }
    merge(vec_channels, modified_image);

    cvtColor(modified_image, modified_image, COLOR_YCrCb2BGR);

    return modified_image;
}

Mat MainWindow::mul(double value){

    Mat modified_image;
    image.copyTo(modified_image);

    cvtColor(modified_image, modified_image, COLOR_BGR2YCrCb);

    vector<Mat> vec_channels;
    split(modified_image, vec_channels);

        for(int j=0;j<vec_channels[0].rows;j++)
        {
              for (int i=0;i<vec_channels[0].cols;i++)
              {
                  vec_channels[0].at<uchar>(j,i) *= value;
              }
        }
    merge(vec_channels, modified_image);

    cvtColor(modified_image, modified_image, COLOR_YCrCb2BGR);

    return modified_image;
}

Mat MainWindow::exponentiation(double value){

    Mat modified_image;
    image.copyTo(modified_image);

    cvtColor(modified_image, modified_image, COLOR_BGR2YCrCb);

    vector<Mat> vec_channels;
    split(modified_image, vec_channels);

        for(int j=0;j<vec_channels[0].rows;j++)
        {
              for (int i=0;i<vec_channels[0].cols;i++)
              {
                  vec_channels[0].at<uchar>(j,i) = 255*qPow(vec_channels[0].at<uchar>(j,i)/255.0,value);
              }
        }
    merge(vec_channels, modified_image);

    cvtColor(modified_image, modified_image, COLOR_YCrCb2BGR);

    return modified_image;
}

Mat MainWindow::negative(){
    Mat modified_image;
    image.copyTo(modified_image);

    cvtColor(modified_image, modified_image, COLOR_BGR2YCrCb);

    vector<Mat> vec_channels;
    split(modified_image, vec_channels);

        for(int j=0;j<vec_channels[0].rows;j++)
        {
              for (int i=0;i<vec_channels[0].cols;i++)
              {
                  vec_channels[0].at<uchar>(j,i) = 255-vec_channels[0].at<uchar>(j,i);
              }
        }
    merge(vec_channels, modified_image);

    cvtColor(modified_image, modified_image, COLOR_YCrCb2BGR);

    return modified_image;
}
Mat MainWindow::logariphmic(){

    Mat modified_image;
    image.copyTo(modified_image);

    cvtColor(modified_image, modified_image, COLOR_BGR2YCrCb);

    vector<Mat> vec_channels;
    split(modified_image, vec_channels);

        double min,max;
        minMaxLoc(modified_image,&min,&max);
        for(int j=0;j<vec_channels[0].rows;j++)
        {
              for (int i=0;i<vec_channels[0].cols;i++)
              {
                  int x = vec_channels[0].at<uchar>(j,i);
                  vec_channels[0].at<uchar>(j,i) = 255*qLn(1+x)/qLn(1+max);
              }
        }
    merge(vec_channels, modified_image);

    cvtColor(modified_image, modified_image, COLOR_YCrCb2BGR);

    return modified_image;
}

void MainWindow::resizeEvent(QResizeEvent *event){
    ui->scrollArea->resize(width(),height()*0.9);
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_horizontalSlider_sliderMoved(int position)
{
    Mat image_add = add(position);
    Mat image_add_hist = buildHistogram(image_add);
    imwrite("modified_images\\image_add.jpg",image_add);
    imwrite("modified_images\\image_add_hist.jpg",image_add_hist);
    ui->label_23->setPixmap(QPixmap("modified_images\\image_add.jpg").scaled(200,200));
    ui->label_24->setPixmap(QPixmap("modified_images\\image_add_hist.jpg").scaled(200,200));
}


void MainWindow::on_horizontalSlider_2_sliderMoved(int position)
{
    Mat image_mul = mul(position/100.0);
    Mat image_mul_hist = buildHistogram(image_mul);
    imwrite("modified_images\\image_mul.jpg",image_mul);
    imwrite("modified_images\\image_mul_hist.jpg",image_mul_hist);
    ui->label_29->setPixmap(QPixmap("modified_images\\image_mul.jpg").scaled(200,200));
    ui->label_28->setPixmap(QPixmap("modified_images\\image_mul_hist.jpg").scaled(200,200));
}


void MainWindow::on_horizontalSlider_3_sliderMoved(int position)
{
    Mat image_exp = exponentiation(position/100.0);
    Mat image_exp_hist = buildHistogram(image_exp);
    imwrite("modified_images\\image_exp.jpg",image_exp);
    imwrite("modified_images\\image_exp_hist.jpg",image_exp_hist);
    ui->label_32->setPixmap(QPixmap("modified_images\\image_exp.jpg").scaled(200,200));
    ui->label_31->setPixmap(QPixmap("modified_images\\image_exp_hist.jpg").scaled(200,200));
}


void MainWindow::on_a_open_triggered()
{
    filename = QFileDialog::getOpenFileName(this).toStdString();
    image = imread(filename);
    setUpImagesAndHistograms();
}

void MainWindow::setUpImagesAndHistograms(){
    Mat linImage = linearContrast(image);
    Mat linImageHistogram = buildHistogram(linImage);
    imwrite("modified_images\\linImage.jpg",linImage);
    imwrite("modified_images\\linImageHistogram.jpg",linImageHistogram);

    Mat hist_equalized_image = equalImageHist(image);
    Mat hist_equalizedRGB_image = equalImageHistRGB(image);

    Mat histImage = buildHistogram(image);
    Mat histImage2 = buildHistogram(hist_equalized_image);
    Mat histImage3 = buildHistogram(hist_equalizedRGB_image);

    Mat negative_image = negative();
    Mat negative_image_hist = buildHistogram(hist_equalizedRGB_image);

    Mat logariphmic_image = logariphmic();
    Mat logariphmic_image_hist = buildHistogram(logariphmic_image);

    on_horizontalSlider_2_sliderMoved(100);
    on_horizontalSlider_3_sliderMoved(100);

        imwrite("modified_images\\hist_equalized_image.jpg",hist_equalized_image);
        imwrite("modified_images\\orig_hist.jpg",histImage);
        imwrite("modified_images\\final_hist.jpg",histImage2);
        imwrite("modified_images\\hist_equalizedRGB_image.jpg",hist_equalizedRGB_image);
        imwrite("modified_images\\histImage3.jpg",histImage3);
        imwrite("modified_images\\logariphmic_image.jpg",logariphmic_image);
        imwrite("modified_images\\logariphmic_image_hist.jpg",logariphmic_image_hist);
        imwrite("modified_images\\negative_image.jpg",negative_image);
        imwrite("modified_images\\negative_image_hist.jpg",negative_image_hist);
        ui->label_2->setPixmap(QPixmap(QString::fromStdString(filename)).scaled(200,200));
        ui->label_11->setPixmap(QPixmap(QString::fromStdString(filename)).scaled(200,200));
        ui->label_23->setPixmap(QPixmap(QString::fromStdString(filename)).scaled(200,200));
        ui->label_3->setPixmap(QPixmap("modified_images\\hist_equalized_image.jpg").scaled(200,200));
        ui->label_4->setPixmap(QPixmap("modified_images\\orig_hist.jpg.").scaled(200,200));
        ui->label_13->setPixmap(QPixmap("modified_images\\orig_hist.jpg.").scaled(200,200));
        ui->label_24->setPixmap(QPixmap("modified_images\\orig_hist.jpg.").scaled(200,200));
        ui->label_5->setPixmap(QPixmap("modified_images\\final_hist.jpg.").scaled(200,200));
        ui->label_10->setPixmap(QPixmap("modified_images\\linImage.jpg").scaled(200,200));
        ui->label_12->setPixmap(QPixmap("modified_images\\linImageHistogram.jpg").scaled(200,200));
        ui->label_16->setPixmap(QPixmap("modified_images\\hist_equalizedRGB_image.jpg").scaled(200,200));
        ui->label_17->setPixmap(QPixmap("modified_images\\histImage3.jpg").scaled(200,200));
        ui->label_19->setPixmap(QPixmap("modified_images\\hist_equalized_image.jpg").scaled(200,200));
        ui->label_20->setPixmap(QPixmap("modified_images\\final_hist.jpg").scaled(200,200));
        ui->label_26->setPixmap(QPixmap("modified_images\\negative_image.jpg").scaled(200,200));
        ui->label_25->setPixmap(QPixmap("modified_images\\negative_image_hist.jpg").scaled(200,200));
        ui->label_34->setPixmap(QPixmap("modified_images\\logariphmic_image.jpg").scaled(200,200));
        ui->label_35->setPixmap(QPixmap("modified_images\\logariphmic_image_hist.jpg").scaled(200,200));
}

