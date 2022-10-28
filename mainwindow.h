#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <QDebug>
#include <QVector>

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
    void on_horizontalSlider_sliderMoved(int position);

    void on_horizontalSlider_2_sliderMoved(int position);

    void on_horizontalSlider_3_sliderMoved(int position);

    void on_a_open_triggered();

private:
    Ui::MainWindow *ui;
    cv::Mat image;

    void setUpImagesAndHistograms();
    std::string filename;
    cv::Mat buildHistogram(cv::Mat image);
    cv::Mat equalImageHist(cv::Mat image);
    cv::Mat equalImageHistRGB(cv::Mat image);
    cv::Mat add(int value = 0);
    cv::Mat negative();
    cv::Mat mul(double value = 0);
    cv::Mat exponentiation(double value = 0);
    cv::Mat logariphmic();

    cv::Mat linearContrast(cv::Mat image);
    void resizeEvent(QResizeEvent *event);
};
#endif // MAINWINDOW_H
