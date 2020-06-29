from PyQt5.QtWidgets import QLabel, QSizePolicy, QMenu
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

from .Tag import Tag

class ImageView(QLabel):
    def __init__(self, width, editable=False):
      super(QLabel, self).__init__()
      self.path= None
      self.image_id= None
      self.setAlignment(Qt.AlignTop);
      self.corner_y= 0
      self.tags= []
      self.prefered_width= width
      self.editable= editable
      #self.resizeEvent=self.onResize
      #print("Parent", parent, parent.width(), parent.height())
      #self.setGeometry(0,0,410,307)
      self.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)) 

    def onResize(self, event):
      print(self, self.width(), self.height(), event.oldSize())

    def setSample(self, image_id, path, tags):
      self.clear()
      self.path= path
      self.image_id= image_id
      self.icon= QPixmap(self.path)#,Qt.KeepAspectRatio);
      w = self.prefered_width
      h = w*(self.icon.height()/self.icon.width())#self.height()
      #print(w,h,"w,h")
      self.icon= self.icon.scaled(w,h)
      self.setPixmap(self.icon)
      for label in tags:
          self.tags.append(Tag(self, label.id, label.category, [label.bbox_X1, label.bbox_Y1, label.bbox_X2, label.bbox_Y2], self.editable))

    def clear(self):  
      if self.path is not None:
        self.path= None
        for t in self.tags:
          t.setParent(None)
        self.tags.clear()
        self.icon= QPixmap()
        self.setPixmap(self.icon)

    def getFinal(self):
      tags=[]
      for tag in self.tags:
        #print("final",tag.getFinal())
        tags.append((tag.label,tag.getFinal()))
      return (self.image_id, tags)

    def contextMenuEvent(self, event):
       
      menu = QMenu(self)
      if self.editable:
        quitAction = menu.addAction("Add Tag")
        action = menu.exec_(self.mapToGlobal(event.pos()))
        if action == quitAction:
          self.tags.append(Tag(self,Category.get(-1),[0,0,0.1,0.1],True, Qt.red))
          #pass
      else:
        resetAction = menu.addAction("Reset Image")
        action = menu.exec_(self.mapToGlobal(event.pos()))
        if action == resetAction:
          #self.tags.append(TContainer(self,Category.get(-1),[0,0,0.1,0.1],True, Qt.red))
          pass

