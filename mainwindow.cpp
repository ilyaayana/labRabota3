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
    , ui(new Ui::MainWindow),info1(this),info2(this),actions(this),ColorMode(COLOR),task(EQUALIZE_CONTRAST)
{
    ui->setupUi(this);
    actions.addAction(ui->a_hist_and_contrast);
    actions.addAction(ui->a_hist_HSV_RGB);
    actions.addAction(ui->a_add);
    actions.addAction(ui->a_multiply);
    actions.addAction(ui->a_power);
    actions.addAction(ui->a_logariphmic);
    actions.addAction(ui->a_negative);
    for(auto action:actions.actions())
        action->setCheckable(true);
    ui->a_hist_and_contrast->setChecked(true);

    actions.setEnabled(false);

    info1.setWindowTitle("Процесс");
    info2.setWindowTitle("Процесс");

    modifiedImage1 = Mat(30,30,CV_16SC3);
    modifiedImage2 = Mat(30,30,CV_16SC3);

    ui->scrollArea->setVisible(false);
    chooseFilesText = new QLabel("Вы не выбрали файл. Для выбора файла выберите Файл->Открыть",this);
    chooseFilesText->show();

}

void MainWindow::resizeEvent(QResizeEvent *event){
    chooseFilesText->setGeometry(width()*0.18,height()*0.2,700,300);
    ui->scrollArea->resize(width(),height()*0.9);
}

void MainWindow::setUpInterface(){

    actions.setEnabled(true);
    chooseFilesText->hide();
    ui->scrollArea->setVisible(true);
    if((task == EQUALIZE_CONTRAST) || (task == EQUALIZE_RGB_HSV)){
        ui->w_original->setVisible(true);
        ui->w_modified1->setVisible(true);
        ui->horizontalWidget->setVisible(false);
        ui->verticalLayout_9->addWidget(ui->lb_hist3);
        size = 400;
    }
    else{
        ui->w_original->setVisible(false);
        ui->w_modified1->setVisible(false);
        ui->horizontalWidget->setVisible(true);
        ui->horizontalLayout_4->addWidget(ui->lb_hist3);        
        ui->pushButton_3->setText("Сменить цветовой режим");
        size = 600;
    }
    switch(task){
        case EQUALIZE_CONTRAST:
            ui->lb_header->setText("Эквализация гистограммы и Линейное контрастирование");
            ui->pushButton_3->setText("Линейное контрастирование");
            break;
        case EQUALIZE_RGB_HSV:
            ui->lb_header->setText("Эквализация гистограммы");
            ui->pushButton_3->setText("Эквализация гистограммы(RGB)");
            break;
        case ADD:
            ui->lb_header->setText("Добавление значения");
            ui->horizontalSlider->setMaximum(100);
            ui->doubleSpinBox->setMaximum(100);
            break;
        case MULTIPLY:
            ui->lb_header->setText("Умножение на значение");
            ui->horizontalSlider->setMaximum(30);
            ui->doubleSpinBox->setMaximum(3);
            break;
        case POWER:
            ui->lb_header->setText("Возведение в степень");
            ui->horizontalSlider->setMaximum(30);
            ui->doubleSpinBox->setMaximum(3);
            break;
        case LOGARIPHMIC:
            ui->lb_header->setText("Логарифмическое преобразование");
            ui->horizontalWidget->setVisible(false);
            break;
        case NEGATIVE:
            ui->lb_header->setText("Негатив");
            ui->horizontalWidget->setVisible(false);
            break;
    }
}

Mat MainWindow::equalImageHist(Mat image){
    Mat hist_equalized_image;

    cvtColor(image, hist_equalized_image, COLOR_BGR2YCrCb);

    vector<Mat> vec_channels;
    split(hist_equalized_image, vec_channels);

    equalizeHist(vec_channels[0], vec_channels[0]);

    merge(vec_channels, hist_equalized_image);

    cvtColor(hist_equalized_image, hist_equalized_image, COLOR_YCrCb2BGR);

    info1.setText("1.Конвертация изображения из пространства BGR(используемый openCV по умолчанию) в пространство YCrCb\n"
                  "2.Разделение изображения на составляющие с целью извлечения компонента яркости\n"
                  "3.Эквализация гистограммы компонента яркости\n"
                  "4.Объединение полученного изображения и конвертация в пространство BGR");

    return hist_equalized_image;
}
Mat MainWindow::equalImageHistRGB(Mat image){

    Mat modified_image;
    image.copyTo(modified_image);

    vector<Mat> vec_channels;
    split(modified_image, vec_channels);

    for(int i = 0; i<3;i++)
       equalizeHist(vec_channels[i], vec_channels[i]);

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
Mat MainWindow::buildHistogramGray(Mat src){

    Mat image;
    src.copyTo(image);
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;
    Mat hist;
    calcHist( &image, 1, 0, Mat(), hist, 1, &histSize, histRange, uniform, accumulate );
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );
    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ),
              Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
              Scalar( 255, 255, 255), 2, 8, 0  );
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

    info2.setText("1.Конвертация изображения из пространства BGR(используемый openCV по умолчанию) в пространство YCrCb\n"
                  "2.Разделение изображения на составляющие с целью извлечения компонента яркости\n"
                  "3.Вычисление реального диапазона яркостей исходного изображения:fmin = " + QString::number(min) + ",f max = "+ QString::number(max) +"\n"
                  "4.Применение к каждому элементу изображения преобразования по формуле:(f(m,n)-fmin)*255/(fmax-fmin)\n"
                  "5.Объединение полученного изображения и конвертация в пространство BGR");
    return modified_image;
}

Mat MainWindow::add(int value){

    Mat modified_image;
    originalImage.copyTo(modified_image);

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
    originalImage.copyTo(modified_image);

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
    originalImage.copyTo(modified_image);

    cvtColor(modified_image, modified_image, COLOR_BGR2YCrCb);

    vector<Mat> vec_channels;
    split(modified_image, vec_channels);

    double min,max;
    minMaxLoc(modified_image,&min,&max);

        for(int j=0;j<vec_channels[0].rows;j++)
        {
              for (int i=0;i<vec_channels[0].cols;i++)
              {
                  vec_channels[0].at<uchar>(j,i) = 255*qPow(vec_channels[0].at<uchar>(j,i)/max,value);
              }
        }
    merge(vec_channels, modified_image);

    cvtColor(modified_image, modified_image, COLOR_YCrCb2BGR);

    return modified_image;
}

Mat MainWindow::negative(){
    Mat modified_image;
    originalImage.copyTo(modified_image);

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
    originalImage.copyTo(modified_image);

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

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::modifyImage(double value)
{
    switch(task){
        case ADD:
            modifiedImage2 = add(value);
            break;
        case MULTIPLY:
            value/=10;
            modifiedImage2 = mul(value);
            break;
        case POWER:
            value/=10;
            modifiedImage2 = exponentiation(value);
            break;
        case LOGARIPHMIC:
            modifiedImage2 = logariphmic();
            break;
        case NEGATIVE:
            modifiedImage2 = negative();
            break;
        default:
            return;
    }

    Mat histModifiedImage2;
    if(ColorMode == COLOR)
        histModifiedImage2 = buildHistogram(modifiedImage2);
    else
    {
        cvtColor(modifiedImage2, modifiedImage2, COLOR_BGR2GRAY);
        histModifiedImage2 = buildHistogramGray(modifiedImage2);
    }

    imwrite("modified_images\\modifiedImage2.jpg",modifiedImage2);
    imwrite("modified_images\\histModifiedImage2.jpg",histModifiedImage2);
    ui->lb_Image3->setPixmap(QPixmap("modified_images\\modifiedImage2.jpg").scaled(size,size,Qt::KeepAspectRatio));
    ui->lb_hist3->setPixmap(QPixmap("modified_images\\histModifiedImage2.jpg").scaled(size,size,Qt::KeepAspectRatio));
}


void MainWindow::on_a_open_triggered()
{
    filename = QFileDialog::getOpenFileName(this).toStdString();
    if(filename.empty())
        return;
    setUpInterface();
    setUpImagesAndHistograms();
    modifyImage();
}

void MainWindow::setUpImagesAndHistograms(){


    originalImage = imread(filename);

    switch(task)
    {
        case EQUALIZE_CONTRAST:
            modifiedImage1 = equalImageHist(originalImage);
            modifiedImage2 = linearContrast(originalImage);
            break;
        case EQUALIZE_RGB_HSV:
            modifiedImage1 = equalImageHist(originalImage);
            modifiedImage2 = equalImageHistRGB(originalImage);
            break;
        default:
            return;
    }


    Mat histOriginalImage, histModifiedImage1, histModifiedImage2;
    if(ColorMode == GRAY)
    {
         cvtColor(originalImage, originalImage, COLOR_BGR2GRAY);
         cvtColor(modifiedImage1, modifiedImage1, COLOR_BGR2GRAY);
         cvtColor(modifiedImage2, modifiedImage2, COLOR_BGR2GRAY);
         histOriginalImage = buildHistogramGray(originalImage);
         histModifiedImage1 = buildHistogramGray(modifiedImage1);
         histModifiedImage2 = buildHistogramGray(modifiedImage2);
    }
    else
    {
         histOriginalImage = buildHistogram(originalImage);
         histModifiedImage1 = buildHistogram(modifiedImage1);
         histModifiedImage2 = buildHistogram(modifiedImage2);
    }

    imwrite("modified_images\\orig_image.jpg",originalImage);
    imwrite("modified_images\\modifiedImage1.jpg",modifiedImage1);
    imwrite("modified_images\\modifiedImage2.jpg",modifiedImage2);
    imwrite("modified_images\\histOriginalImage.jpg",histOriginalImage);
    imwrite("modified_images\\histModifiedImage1.jpg",histModifiedImage1);
    imwrite("modified_images\\histModifiedImage2.jpg",histModifiedImage2);


    ui->lb_Image1->setPixmap(QPixmap("modified_images\\orig_image.jpg").scaled(size,size,Qt::KeepAspectRatio));
    ui->lb_Image2->setPixmap(QPixmap("modified_images\\modifiedImage1.jpg").scaled(size,size,Qt::KeepAspectRatio));
    ui->lb_Image3->setPixmap(QPixmap("modified_images\\modifiedImage2.jpg").scaled(size,size,Qt::KeepAspectRatio));
    ui->lb_hist1->setPixmap(QPixmap("modified_images\\histOriginalImage.jpg").scaled(size,size,Qt::KeepAspectRatio));
    ui->lb_hist2->setPixmap(QPixmap("modified_images\\histModifiedImage1.jpg").scaled(size,size,Qt::KeepAspectRatio));
    ui->lb_hist3->setPixmap(QPixmap("modified_images\\histModifiedImage2.jpg").scaled(size,size,Qt::KeepAspectRatio));
}


void MainWindow::on_pushButton_clicked()
{
    info1.exec();
}


void MainWindow::on_pushButton_3_clicked()
{
    if((task == EQUALIZE_CONTRAST) || (task == EQUALIZE_RGB_HSV))
        info2.exec();
    else{
       ColorMode = ColorMode == COLOR ? GRAY: COLOR;
       modifyImage(ui->horizontalSlider->value());
    }
}


void MainWindow::on_pb_original_clicked()
{
    ColorMode = ColorMode == COLOR ? GRAY: COLOR;
    setUpImagesAndHistograms();
}


void MainWindow::on_horizontalSlider_sliderMoved(int position)
{
    if(task == ADD)
        ui->doubleSpinBox->setValue(position);
    else
        ui->doubleSpinBox->setValue(position/10.0);
     modifyImage(position);
}

void MainWindow::on_doubleSpinBox_editingFinished()
{
    if(task == ADD)
        ui->horizontalSlider->setValue(ui->doubleSpinBox->value());
    else
        ui->horizontalSlider->setValue(ui->doubleSpinBox->value()*10);
    modifyImage(ui->horizontalSlider->value());
}

void MainWindow::on_a_hist_and_contrast_triggered()
{
    ColorMode = COLOR;
    task = EQUALIZE_CONTRAST;
    setUpInterface();
    setUpImagesAndHistograms();
}
void MainWindow::on_a_hist_HSV_RGB_triggered()
{
    ColorMode = COLOR;
    task = EQUALIZE_RGB_HSV;
    setUpInterface();
    setUpImagesAndHistograms();
}


void MainWindow::on_a_add_triggered()
{
    ColorMode = COLOR;
    task = ADD;
    setUpInterface();
    setUpImagesAndHistograms();
}


void MainWindow::on_a_multiply_triggered()
{
    ColorMode =COLOR;
    task = MULTIPLY;
    setUpInterface();
    setUpImagesAndHistograms();
    modifyImage();
}


void MainWindow::on_a_power_triggered()
{
    ColorMode = COLOR;
    task = POWER;
    setUpInterface();
    setUpImagesAndHistograms();
    modifyImage();
}


void MainWindow::on_a_logariphmic_triggered()
{
    ColorMode =COLOR;
    task = LOGARIPHMIC;
    setUpInterface();
    setUpImagesAndHistograms();
    modifyImage();
}


void MainWindow::on_a_negative_triggered()
{
    ColorMode = COLOR;
    task = NEGATIVE;
    setUpInterface();
    setUpImagesAndHistograms();
    modifyImage();
}

