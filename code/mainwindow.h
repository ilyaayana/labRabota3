#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <QDebug>
#include <QVector>
#include <QLabel>
#include <QMessageBox>
#include <QActionGroup>

enum Mode{
    GRAY,
    COLOR
};

enum Task{
    EQUALIZE_CONTRAST,
    EQUALIZE_RGB_HSV,
    ADD,
    MULTIPLY,
    POWER,
    LOGARIPHMIC,
    NEGATIVE
};

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
//    void on_horizontalSlider_sliderMoved(int position);
//    void on_horizontalSlider_2_sliderMoved(int position);
//    void on_horizontalSlider_3_sliderMoved(int position);
    void on_a_open_triggered();

    void on_pushButton_clicked();

    void on_pushButton_3_clicked();

    void on_pb_original_clicked();

    void on_a_hist_and_contrast_triggered();

    void on_a_hist_HSV_RGB_triggered();

    void on_a_add_triggered();

    void on_horizontalSlider_sliderMoved(int position);

    void on_doubleSpinBox_editingFinished();

    void on_a_multiply_triggered();

    void on_a_power_triggered();

    void on_a_logariphmic_triggered();

    void on_a_negative_triggered();

private:
    Ui::MainWindow *ui;
    cv::Mat originalImage;
    cv::Mat modifiedImage1,modifiedImage2;

    QLabel* chooseFilesText;

    Mode ColorMode;
    Task task;

    void setUpImagesAndHistograms();
    std::string filename;
    cv::Mat buildHistogram(cv::Mat image);
    cv::Mat buildHistogramGray(cv::Mat image);
    cv::Mat equalImageHist(cv::Mat image);
    cv::Mat equalImageHistRGB(cv::Mat image);
    cv::Mat add(int value = 0);
    cv::Mat negative();
    cv::Mat mul(double value = 0);
    cv::Mat exponentiation(double value = 0);
    cv::Mat logariphmic();
    cv::Mat linearContrast(cv::Mat image);

    void modifyImage(double value = 0);

    void setUpInterface();
    void resizeEvent(QResizeEvent *event);

    QActionGroup actions;
    QMessageBox info1,info2;
    int size;


};
#endif // MAINWINDOW_H
