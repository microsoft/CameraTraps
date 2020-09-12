from PyQt5.QtWidgets import QTabWidget, QMainWindow, QApplication, QSizePolicy, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QProgressBar, QMessageBox
from PyQt5.QtCore import Qt
from .DBObjects import *


class SpeciesWidget(QWidget):

    def __init__(self):
        super(SpeciesWidget, self).__init__()
        self.verticalLayout= QVBoxLayout(self)
        self.speciesList= QTableWidget()
        self.speciesList.setRowCount(Category.select().count())
        self.speciesList.setColumnCount(3)
        self.speciesList.verticalHeader().setVisible(False)
        self.speciesList.setHorizontalHeaderItem(0,QTableWidgetItem("ID"))
        self.speciesList.setHorizontalHeaderItem(1,QTableWidgetItem("Name"))
        self.speciesList.setHorizontalHeaderItem(2,QTableWidgetItem("Short Name"))
        self.id_dict={}
        for i, record in enumerate(Category.select().order_by(Category.id)):
            id_item= QTableWidgetItem(str(record.id))
            id_item.setFlags(id_item.flags() ^ Qt.ItemIsEditable)
            self.speciesList.setItem(i,0, id_item)
            self.speciesList.setItem(i,1, QTableWidgetItem(record.name))
            self.speciesList.setItem(i,2, QTableWidgetItem(record.abbr))
            self.id_dict[i]=record.id
        #self.tab4.speciesList.setModel(species)
        #self.tab4.speciesList.setRowHidden(len(species.stringList())-1, True)
        self.insert = QPushButton('Add New' )
        self.delete = QPushButton('Delete Current Row')
        self.save = QPushButton('Save Changes' )
        self.horizontalLayout= QHBoxLayout()
        self.horizontalLayout.addWidget(self.insert) 
        self.horizontalLayout.addWidget(self.delete) 
        self.horizontalLayout.addWidget(self.save) 
        self.verticalLayout.addWidget(self.speciesList)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.insert.clicked.connect(self.addSpecies)
        self.delete.clicked.connect(self.deleteSpecies)
        self.save.clicked.connect(self.syncSpecies)
        #self.speciesList.itemChanged.connect(self.itemChanged)

    def addSpecies(self, event):
      rowPosition = self.speciesList.rowCount()
      self.speciesList.insertRow(rowPosition)
      self.speciesList.setCurrentItem(self.speciesList.item(rowPosition,0))
    
    def validate(self):
      allRows = self.speciesList.rowCount()
      itemsList=[]
      id_set= set()
      name_set= set()
      shortname_set= set()
      for row in range(allRows):
          id_item= self.speciesList.item(row,0)
          rowid = int(id_item.text())
          rowname = self.speciesList.item(row,1).text()
          rowshortname = self.speciesList.item(row,2).text()
          if rowid is None or rowid in id_set:
              return "There is an issue with %s at row %d"%(rowid,rowid), itemsList
          if rowname is None or rowname in name_set:
              return "There is an issue with %s at row %d"%(rowname,rowid), itemsList
          if rowshortname is None or rowshortname in shortname_set or len(rowshortname)>2:
              return "There is an issue with %s at row %d"%(rowshortname,rowid), itemsList
          id_set.add(rowid)
          name_set.add(rowname)
          shortname_set.add(rowshortname)
          itemsList.append((rowid,rowname,rowshortname,row, id_item))
      return None,itemsList

    def syncSpecies(self, event):
        val, itemsList= self.validate()
        if val is None:
            for x in itemsList:
                species= Category.get_or_create(id=x[0], name= x[1], abbr=x[2])
                x[4].setFlags(x[4].flags() ^ Qt.ItemIsEditable)
            Category.delete().where(Category.id.not_in([x[0] for x in itemsList])).execute()
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("The changes are successfully saved!")
            msg.setWindowTitle("Save Completed")
            msg.setInformativeText(val)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("The entered data contain an invalid item")
            msg.setWindowTitle("Invalid Data")
            msg.setDetailedText(val)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

    def deleteSpecies(self,event):
      selected = self.speciesList.currentRow()
      self.speciesList.removeRow(selected)
      
