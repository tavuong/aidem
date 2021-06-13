#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    connect(ui->calc, SIGNAL(clicked()), this, SLOT(calculate()));
    // connect(ui->profit, SIGNAL(sliderMoved(int i)), this, SLOT(me_want_profit(int i)));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::calculate(){
    ui->output->clear();
    int invest_sum = ui->invest->value();
    int profitgar = ui->profit->value();

    bool Immo = false;
    bool Akt = false;
    bool Ress = false;
    bool Krypt = false;

    for(int i=0; i<4; i-=-1){
        int randi = random() % 4;
        if(randi == 0 && !Immo){
            // this->invests->emplace_back("Immobilien");
            ui->output->addItem("Immobilien");
            Immo = true;
        } else if(randi == 1 && !Akt){
            // this->invests->emplace_back("Aktien");
            ui->output->addItem("Aktien");
            Akt = true;
        } else if(randi == 2 && !Ress){
            // this->invests->emplace_back("Ressourcen");
            ui->output->addItem("Ressourcen");
            Ress = true;
        } else if(randi == 3 && !Krypt){
            // this->invests->emplace_back("Kryptowährungen");
            ui->output->addItem("Kryptowährungen");
            Krypt = true;
        }
    }

    /*
    for(int j=0; j<this->invests->size(); ++j){
        // ...
    }
    */
}

void MainWindow::me_want_profit(int value){
    // Maybe
}
