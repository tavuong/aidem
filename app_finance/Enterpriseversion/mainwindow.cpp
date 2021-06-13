#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->outlabel->setVisible(false);
    connect(ui->calc, SIGNAL(clicked()), this, SLOT(do_invest()));
    connect(ui->reset, SIGNAL(clicked()), this, SLOT(hard_reset()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::do_invest(){
    ui->outlabel->setVisible(true);
}

void MainWindow::hard_reset(){
    ui->outlabel->setVisible(false);
    ui->invest->setValue(100000);
    ui->invtime->setValue(100);
    ui->age->setValue(100);
    ui->expectwin->setValue(100000);
    ui->Aktcheck->setCheckState(Qt::Unchecked);
    ui->Immocheck->setCheckState(Qt::Unchecked);
    ui->Resscheck->setCheckState(Qt::Unchecked);
    ui->Kryptcheck->setCheckState(Qt::Unchecked);
}
