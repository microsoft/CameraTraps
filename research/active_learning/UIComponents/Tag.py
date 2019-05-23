import sys
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QWidget, QMenu, QCheckBox, QLabel, QApplication
from PyQt5.QtCore import Qt,QPoint, pyqtSignal, QRect, QSize
from PyQt5.QtGui import QColor, QCursor, QPainterPath, QBrush, QPen
from enum import Enum

class Mode(Enum):
    NONE = 0,
    MOVE = 1,
    RESIZETL = 2,
    RESIZET = 3,
    RESIZETR = 4,
    RESIZER = 5,
    RESIZEBR = 6,
    RESIZEB = 7,
    RESIZEBL = 8,
    RESIZEL = 9

class Tag(QWidget):
    """ allow to move and resize by user"""
    menu = None
    mode = Mode.NONE
    position = None
    inFocus = pyqtSignal(bool)
    outFocus = pyqtSignal(bool)
    newGeometry = pyqtSignal(QRect)

    def __init__(self, parent,detection_id, label, bbox, editable= False, color= None):
        super().__init__(parent=parent)
        #combo.setEnabled(not finalized)
        self.menu = QMenu(parent=self, title='menu')
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.move(QPoint(0,0))

        self.m_infocus = True
        self.m_showMenu = False
        self.m_isEditing = editable
        self.installEventFilter(parent)
        self.bbox= bbox
        self.detection_id= detection_id
        
        x= round(bbox[1]*self.parentWidget().pixmap().width())
        y= round(bbox[0]*self.parentWidget().pixmap().height())
        w= round(bbox[3]*self.parentWidget().pixmap().width()-x)
        h= round(bbox[2]*self.parentWidget().pixmap().height()- y)
        self.setGeometry(x,y,w,h)
        self.tik= QCheckBox(self)
        self.tik.move(0,0)
        self.title= QLabel(self)
        self.title.move(15,0)
        self.title.setStyleSheet("QLabel {  color : white; font-weight: bold; }")
        if not editable:
          self.tik.hide()
          self.title.move(0,0)
        self.updateLabel(label)
        self.pen= QPen()
        self.pen.setStyle(Qt.SolidLine)
        self.pen.setWidth(3)
        self.pen.setBrush(self.color)

        #print(self.bbox)
        self.setVisible(True)
        self.setAutoFillBackground(False)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setFocus()

    """def resizeEvent(self, event):
        print("resize invalidate",event.oldSize())
        self.valid= False

    def moveEvent(self, event):
        print("move invalidate", event.oldPos())
        self.valid= False"""
    def updateLabel(self, label):
        self.label= label
        self.title.setText(self.label.name)
        if self.width()<40:
          self.title.setText(self.label.name[0:2])
          if self.m_isEditing:
            self.title.move(0,15)
        if self.label.id==-1:
          self.color=Qt.red
        else:
          self.color=Qt.green
        if hasattr(self,'pen'):
          self.pen.setBrush(self.color)
        self.tik.setCheckState(False)
        #self.addWidget(self.child)
        self.update()
       
    def getFinal(self):
        y= self.x()/self.parentWidget().pixmap().width()
        x= self.y()/self.parentWidget().pixmap().height()
        h= (self.x()+self.width())/self.parentWidget().pixmap().width()
        w= (self.y()+self.height())/self.parentWidget().pixmap().height()
        if abs(x-self.bbox[0])<0.01 and abs(x-self.bbox[0])<0.01 and abs(x-self.bbox[0])<0.01 and abs(x-self.bbox[0])<0.01:
          return self.detection_id, (self.bbox[0],self.bbox[1],self.bbox[2],self.bbox[3])
        else:
          return self.detection_id, (x,y,w,h)

    def speciesChanged(self, text):
        if text=='Add New':
          self.parentWidget().parentWidget().parentWidget().parentWidget().setCurrentIndex(2)
        else:
          self.label= text

    """def setChildWidget(self, cWidget):
        if cWidget:
            self.childWidget = cWidget
            self.childWidget.setParent(self)
            self.childWidget.setMinimumSize(70,20)
            self.childWidget.move(0,0)"""

    def popupShow(self, pt: QPoint):
        if self.menu.isEmpty:
            return
        global_ = self.mapToGlobal(pt)
        self.m_showMenu = True
        self.menu.exec(global_)
        self.m_showMenu = False

    def contextMenuEvent(self, event):
       
      menu = QMenu(self)
      quitAction = menu.addAction("Delete Tag")
      action = menu.exec_(self.mapToGlobal(event.pos()))
      if action == quitAction:
        self.parentWidget().tags.remove(self)
        self.setParent(None)

    def focusInEvent(self, a0: QtGui.QFocusEvent):
        self.m_infocus = True
        p = self.parentWidget()
        p.installEventFilter(self)
        p.repaint()
        self.inFocus.emit(True)

    def focusOutEvent(self, a0: QtGui.QFocusEvent):
        if not self.m_isEditing:
            return
        if self.m_showMenu:
            return
        self.mode = Mode.NONE
        self.outFocus.emit(False)
        self.m_infocus = False

    def paintEvent(self, e: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        rect = e.rect()
        rect.adjust(0,0,-1,-1)
        painter.setPen(self.pen)
        painter.drawRect(rect)

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        self.position = QPoint(e.globalX() - self.geometry().x(), e.globalY() - self.geometry().y())
        if not self.m_isEditing:
            return
        if not self.m_infocus:
            return
        if not e.buttons() and QtCore.Qt.LeftButton:
            self.setCursorShape(e.pos())
            return
        if e.button() == QtCore.Qt.RightButton:
            self.popupShow(e.pos())
            e.accept()

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if not self.m_isEditing: return
        if e.key() == QtCore.Qt.Key_Delete:
            self.deleteLater()
        # Moving container with arrows
        if QApplication.keyboardModifiers() == QtCore.Qt.ControlModifier:
            newPos = QPoint(self.x(), self.y())
            if e.key() == QtCore.Qt.Key_Up:
                newPos.setY(newPos.y() - 1)
            if e.key() == QtCore.Qt.Key_Down:
                newPos.setY(newPos.y() + 1)
            if e.key() == QtCore.Qt.Key_Left:
                newPos.setX(newPos.x() - 1)
            if e.key() == QtCore.Qt.Key_Right:
                newPos.setX(newPos.x() + 1)
            self.move(newPos)

        if QApplication.keyboardModifiers() == QtCore.Qt.ShiftModifier:
            if e.key() == QtCore.Qt.Key_Up:
                self.resize(self.width(), self.height() - 1)
            if e.key() == QtCore.Qt.Key_Down:
                self.resize(self.width(), self.height() + 1)
            if e.key() == QtCore.Qt.Key_Left:
                self.resize(self.width() - 1, self.height())
            if e.key() == QtCore.Qt.Key_Right:
                self.resize(self.width() + 1, self.height())
        self.newGeometry.emit(self.geometry())


    def setCursorShape(self, e_pos: QPoint):
        diff = 3
        # Left - Bottom

        if (((e_pos.y() > self.y() + self.height() - diff) and # Bottom
            (e_pos.x() < self.x() + diff)) or # Left
        # Right-Bottom
        ((e_pos.y() > self.y() + self.height() - diff) and # Bottom
        (e_pos.x() > self.x() + self.width() - diff)) or # Right
        # Left-Top
        ((e_pos.y() < self.y() + diff) and # Top
        (e_pos.x() < self.x() + diff)) or # Left
        # Right-Top
        (e_pos.y() < self.y() + diff) and # Top
        (e_pos.x() > self.x() + self.width() - diff)): # Right
            # Left - Bottom
            if ((e_pos.y() > self.y() + self.height() - diff) and # Bottom
            (e_pos.x() < self.x()
                + diff)): # Left
                self.mode = Mode.RESIZEBL
                self.setCursor(QCursor(QtCore.Qt.SizeBDiagCursor))
                # Right - Bottom
            if ((e_pos.y() > self.y() + self.height() - diff) and # Bottom
            (e_pos.x() > self.x() + self.width() - diff)): # Right
                self.mode = Mode.RESIZEBR
                self.setCursor(QCursor(QtCore.Qt.SizeFDiagCursor))
            # Left - Top
            if ((e_pos.y() < self.y() + diff) and # Top
            (e_pos.x() < self.x() + diff)): # Left
                self.mode = Mode.RESIZETL
                self.setCursor(QCursor(QtCore.Qt.SizeFDiagCursor))
            # Right - Top
            if ((e_pos.y() < self.y() + diff) and # Top
            (e_pos.x() > self.x() + self.width() - diff)): # Right
                self.mode = Mode.RESIZETR
                self.setCursor(QCursor(QtCore.Qt.SizeBDiagCursor))
        # check cursor horizontal position
        elif ((e_pos.x() < self.x() + diff) or # Left
            (e_pos.x() > self.x() + self.width() - diff)): # Right
            if e_pos.x() < self.x() + diff: # Left
                self.setCursor(QCursor(QtCore.Qt.SizeHorCursor))
                self.mode = Mode.RESIZEL
            else: # Right
                self.setCursor(QCursor(QtCore.Qt.SizeHorCursor))
                self.mode = Mode.RESIZER
        # check cursor vertical position
        elif ((e_pos.y() > self.y() + self.height() - diff) or # Bottom
            (e_pos.y() < self.y() + diff)): # Top
            if e_pos.y() < self.y() + diff: # Top
                self.setCursor(QCursor(QtCore.Qt.SizeVerCursor))
                self.mode = Mode.RESIZET
            else: # Bottom
                self.setCursor(QCursor(QtCore.Qt.SizeVerCursor))
                self.mode = Mode.RESIZEB
        else:
            self.setCursor(QCursor(QtCore.Qt. ArrowCursor))
            self.mode = Mode.MOVE


    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        QWidget.mouseReleaseEvent(self, e)

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        QWidget.mouseMoveEvent(self, e)
        if not self.m_isEditing:
            return
        if not self.m_infocus:
            return
        if not e.buttons() and QtCore.Qt.LeftButton:
            p = QPoint(e.x() + self.geometry().x(), e.y() + self.geometry().y())
            self.setCursorShape(p)
            return

        if (self.mode == Mode.MOVE or self.mode == Mode.NONE) and e.buttons() and QtCore.Qt.LeftButton:
            toMove = e.globalPos() - self.position
            if toMove.x() < 0:return
            if toMove.y() < 0:return
            if toMove.x() > self.parentWidget().width() - self.width(): return
            self.move(toMove)
            self.newGeometry.emit(self.geometry())
            self.parentWidget().repaint()
            return
        if (self.mode != Mode.MOVE) and e.buttons() and QtCore.Qt.LeftButton:
            if self.mode == Mode.RESIZETL: # Left - Top
                newwidth = e.globalX() - self.position.x() - self.geometry().x()
                newheight = e.globalY() - self.position.y() - self.geometry().y()
                toMove = e.globalPos() - self.position
                self.resize(self.geometry().width() - newwidth, self.geometry().height() - newheight)
                self.move(toMove.x(), toMove.y())
            elif self.mode == Mode.RESIZETR: # Right - Top
                newheight = e.globalY() - self.position.y() - self.geometry().y()
                toMove = e.globalPos() - self.position
                self.resize(e.x(), self.geometry().height() - newheight)
                self.move(self.x(), toMove.y())
            elif self.mode== Mode.RESIZEBL: # Left - Bottom
                newwidth = e.globalX() - self.position.x() - self.geometry().x()
                toMove = e.globalPos() - self.position
                self.resize(self.geometry().width() - newwidth, e.y())
                self.move(toMove.x(), self.y())
            elif self.mode == Mode.RESIZEB: # Bottom
                self.resize(self.width(), e.y())
            elif self.mode == Mode.RESIZEL: # Left
                newwidth = e.globalX() - self.position.x() - self.geometry().x()
                toMove = e.globalPos() - self.position
                self.resize(self.geometry().width() - newwidth, self.height())
                self.move(toMove.x(), self.y())
            elif self.mode == Mode.RESIZET:# Top
                newheight = e.globalY() - self.position.y() - self.geometry().y()
                toMove = e.globalPos() - self.position
                self.resize(self.width(), self.geometry().height() - newheight)
                self.move(self.x(), toMove.y())
            elif self.mode == Mode.RESIZER: # Right
                self.resize(e.x(), self.height())
            elif self.mode == Mode.RESIZEBR:# Right - Bottom
                self.resize(e.x(), e.y())
            self.parentWidget().repaint()
        self.newGeometry.emit(self.geometry())

