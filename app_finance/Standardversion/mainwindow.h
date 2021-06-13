#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <string>
#include <vector>
#include <random>

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
    void calculate();
    void me_want_profit(int value);

private:
    Ui::MainWindow *ui;
    std::vector<QString> *invests;
    std::vector<QString> *examples;

};
#endif // MAINWINDOW_H
