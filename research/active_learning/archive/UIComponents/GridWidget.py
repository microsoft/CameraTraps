from PyQt5.QtWidgets import QWidget, QGridLayout, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QLineEdit
from PyQt5.QtGui import QIntValidator
import os

from .ImageView import ImageView
from .DBObjects import *

class GridWidget(QWidget):
    def __init__(self, kind, im_width, num_rows=3, num_cols=4, labeler=False):
        super(GridWidget, self).__init__()
        self.num_rows= num_rows
        self.num_cols= num_cols
        self.page= 1
        self.gridLayout= QGridLayout()
        self.kind= kind
        #fullname=str(self.model)
        #self.name= (fullname[fullname.find(":")+2:fullname.find(">")].strip()+'_set').lower()
        self.images=[None]*(self.num_rows*self.num_cols)
        self.updated= True
        self.num_pages = 1
        for i in range(self.num_rows):
          for j in range(self.num_cols):
            index= i*self.num_cols+j
            self.images[index] = ImageView(im_width, labeler)
            self.gridLayout.addWidget(self.images[index], i, j)

        self.next = QPushButton('Next' )
        self.previous = QPushButton('Previous' )
        self.last = QPushButton('Last' )
        self.first = QPushButton('First' )
        self.of = QLabel('of' )
        self.total = QLabel()
        self.current= QLineEdit("1")
        self.current.setValidator(QIntValidator());
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.addWidget(self.first)
        self.horizontalLayout.addWidget(self.previous)
        self.horizontalLayout.addWidget(self.current)
        self.horizontalLayout.addWidget(self.of)
        self.horizontalLayout.addWidget(self.total)
        self.horizontalLayout.addWidget(self.next)
        self.horizontalLayout.addWidget(self.last)
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.addLayout( self.gridLayout, stretch=1)
        self.verticalLayout.addLayout( self.horizontalLayout, stretch=0) 
        self.labeler= labeler
        if labeler:
          self.horizontalLayout2 = QHBoxLayout()
          self.horizontalLayout2.addLayout(self.verticalLayout)
          self.refreshLabeler()
          self.setLayout(self.horizontalLayout2)
        else:
          self.setLayout(self.verticalLayout)
        self.updatePages()
        self.next.clicked.connect(self.doNext)
        self.previous.clicked.connect(self.doPrevious)
        self.last.clicked.connect(self.doLast)
        self.first.clicked.connect(self.doFirst)
        self.current.returnPressed.connect(self.doJump)


    def refreshLabeler(self):
      if hasattr(self, "labelerGrid"):
        for i in reversed(range(self.labelerGrid.count())): 
          self.labelerGrid.itemAt(i).widget().deleteLater()
        
        self.labelerGrid.deleteLater()
        self.labelerGrid.setParent(None)
        self.horizontalLayout2.removeItem(self.labelerGrid)
        del self.labelerGrid
      self.labelerGrid= QGridLayout()
      query= Category.select()
      i=0
      for cat in query:
          button= QPushButton(cat.name)
          button.clicked.connect(self.clickgen(cat))
          self.labelerGrid.addWidget(button, i//2,i%2)
          i+=1
      button= QPushButton("Delete")
      button.clicked.connect(self.delete)
      self.labelerGrid.addWidget(button, i//2,i%2)
      self.horizontalLayout2.addLayout(self.labelerGrid)
      self.update()

    def clickgen(self,cat):
      def labelerClicked(event):
        for i in range(self.num_rows):
          for j in range(self.num_cols):
            index= i*self.num_cols+j
            ex_label= self.images[index]
            for label in ex_label.tags:
              if label.tik.isVisible() and label.tik.checkState():
                label.updateLabel(cat)
      return labelerClicked

    def delete(self,event):
        for i in range(self.num_rows):
          for j in range(self.num_cols):
            index= i*self.num_cols+j
            ex_label= self.images[index]
            for tag in ex_label.tags:
              if tag.tik.isVisible() and tag.tik.checkState():
                tag.setParent(None)
                ex_label.tags.remove(tag)


    def updatePages(self):
      count= Image.select().join(Detection).where(Detection.kind==self.kind.value).count()
      self.num_pages= (count//(self.num_rows*self.num_cols))+1
      self.total.setText(str(self.num_pages))
      self.current.setText(str(self.page))
      if self.page==1:
        self.previous.setEnabled(False)
      else:
        self.previous.setEnabled(True)
      if self.page==self.num_pages:
        self.next.setEnabled(False)
      else:
        self.next.setEnabled(True)

    def doNext(self,event): 
      if self.page<=self.num_pages:
        self.page+=1
        self.updated= True
        self.showCurrentPage()
      self.updatePages()

    def doPrevious(self,event): 
      if self.page>1:
        self.page-=1
        self.updated= True
        self.showCurrentPage()
      self.updatePages()

    def doLast(self,event): 
      self.page= self.num_pages
      self.updated= True
      self.showCurrentPage()
      self.updatePages()

    def doFirst(self,event): 
      self.page=1
      self.updated= True
      self.showCurrentPage()
      self.updatePages()

    def doJump(self):
      val= int(self.current.text()) 
      if val>=1 and val<=self.num_pages:
        self.page= val
        self.updated= True
        self.showCurrentPage()
      else:
        self.current.setText("1")
      self.updatePages()

    def showCurrentPage(self, force=False):
        if self.labeler:
          self.refreshLabeler()
        if self.updated or force:
          #print("Parent", self.parentWidget().width(), self.parentWidget().height())
          self.clear()
          query = Image.select().join(Detection).where(Detection.kind==self.kind.value).paginate(self.page, self.num_rows*self.num_cols)
          #print(self.model,self.name,query.sql())
          index= 0
          for r in query:
            self.images[index].setSample(r.id,os.path.join('/project/evolvingai/mnorouzz/Serengiti/SER',r.file_name),r.detection_set)
            index+=1
          self.updated=False

    def clear(self):
        self.visited= False
        for i in range(self.num_rows):
          for j in range(self.num_cols):
            index= i*self.num_cols+j
            self.images[index].clear()

