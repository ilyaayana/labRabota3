/****************************************************************************
** Meta object code from reading C++ file 'mainwindow.h'
**
** Created by: The Qt Meta Object Compiler version 68 (Qt 6.2.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../laba_3_2/mainwindow.h"
#include <QtGui/qtextcursor.h>
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'mainwindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 68
#error "This file was generated using the moc from 6.2.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_MainWindow_t {
    const uint offsetsAndSize[32];
    char stringdata0[351];
};
#define QT_MOC_LITERAL(ofs, len) \
    uint(offsetof(qt_meta_stringdata_MainWindow_t, stringdata0) + ofs), len 
static const qt_meta_stringdata_MainWindow_t qt_meta_stringdata_MainWindow = {
    {
QT_MOC_LITERAL(0, 10), // "MainWindow"
QT_MOC_LITERAL(11, 19), // "on_a_open_triggered"
QT_MOC_LITERAL(31, 0), // ""
QT_MOC_LITERAL(32, 21), // "on_pushButton_clicked"
QT_MOC_LITERAL(54, 23), // "on_pushButton_3_clicked"
QT_MOC_LITERAL(78, 22), // "on_pb_original_clicked"
QT_MOC_LITERAL(101, 32), // "on_a_hist_and_contrast_triggered"
QT_MOC_LITERAL(134, 27), // "on_a_hist_HSV_RGB_triggered"
QT_MOC_LITERAL(162, 18), // "on_a_add_triggered"
QT_MOC_LITERAL(181, 31), // "on_horizontalSlider_sliderMoved"
QT_MOC_LITERAL(213, 8), // "position"
QT_MOC_LITERAL(222, 32), // "on_doubleSpinBox_editingFinished"
QT_MOC_LITERAL(255, 23), // "on_a_multiply_triggered"
QT_MOC_LITERAL(279, 20), // "on_a_power_triggered"
QT_MOC_LITERAL(300, 26), // "on_a_logariphmic_triggered"
QT_MOC_LITERAL(327, 23) // "on_a_negative_triggered"

    },
    "MainWindow\0on_a_open_triggered\0\0"
    "on_pushButton_clicked\0on_pushButton_3_clicked\0"
    "on_pb_original_clicked\0"
    "on_a_hist_and_contrast_triggered\0"
    "on_a_hist_HSV_RGB_triggered\0"
    "on_a_add_triggered\0on_horizontalSlider_sliderMoved\0"
    "position\0on_doubleSpinBox_editingFinished\0"
    "on_a_multiply_triggered\0on_a_power_triggered\0"
    "on_a_logariphmic_triggered\0"
    "on_a_negative_triggered"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_MainWindow[] = {

 // content:
      10,       // revision
       0,       // classname
       0,    0, // classinfo
      13,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags, initial metatype offsets
       1,    0,   92,    2, 0x08,    1 /* Private */,
       3,    0,   93,    2, 0x08,    2 /* Private */,
       4,    0,   94,    2, 0x08,    3 /* Private */,
       5,    0,   95,    2, 0x08,    4 /* Private */,
       6,    0,   96,    2, 0x08,    5 /* Private */,
       7,    0,   97,    2, 0x08,    6 /* Private */,
       8,    0,   98,    2, 0x08,    7 /* Private */,
       9,    1,   99,    2, 0x08,    8 /* Private */,
      11,    0,  102,    2, 0x08,   10 /* Private */,
      12,    0,  103,    2, 0x08,   11 /* Private */,
      13,    0,  104,    2, 0x08,   12 /* Private */,
      14,    0,  105,    2, 0x08,   13 /* Private */,
      15,    0,  106,    2, 0x08,   14 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,   10,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void MainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<MainWindow *>(_o);
        (void)_t;
        switch (_id) {
        case 0: _t->on_a_open_triggered(); break;
        case 1: _t->on_pushButton_clicked(); break;
        case 2: _t->on_pushButton_3_clicked(); break;
        case 3: _t->on_pb_original_clicked(); break;
        case 4: _t->on_a_hist_and_contrast_triggered(); break;
        case 5: _t->on_a_hist_HSV_RGB_triggered(); break;
        case 6: _t->on_a_add_triggered(); break;
        case 7: _t->on_horizontalSlider_sliderMoved((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 8: _t->on_doubleSpinBox_editingFinished(); break;
        case 9: _t->on_a_multiply_triggered(); break;
        case 10: _t->on_a_power_triggered(); break;
        case 11: _t->on_a_logariphmic_triggered(); break;
        case 12: _t->on_a_negative_triggered(); break;
        default: ;
        }
    }
}

const QMetaObject MainWindow::staticMetaObject = { {
    QMetaObject::SuperData::link<QMainWindow::staticMetaObject>(),
    qt_meta_stringdata_MainWindow.offsetsAndSize,
    qt_meta_data_MainWindow,
    qt_static_metacall,
    nullptr,
qt_incomplete_metaTypeArray<qt_meta_stringdata_MainWindow_t
, QtPrivate::TypeAndForceComplete<MainWindow, std::true_type>
, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<int, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>, QtPrivate::TypeAndForceComplete<void, std::false_type>


>,
    nullptr
} };


const QMetaObject *MainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_MainWindow.stringdata0))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int MainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 13)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 13;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 13)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 13;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
