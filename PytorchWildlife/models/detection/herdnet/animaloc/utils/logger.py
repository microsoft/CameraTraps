__copyright__ = \
    """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 18, 2024
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"


import datetime
import time
import torch
import os
import logging
import csv

# Personnals
from ..utils.torchvision_utils import *
from ..utils.useful_funcs import current_date, get_date_time

class CustomLogger(MetricLogger):
    ''' Custom logger class adapted from torchvision's one '''
    def __init__(self, delimiter='\t', filename=None, work_dir=None, csv: bool=False):
        super(CustomLogger, self).__init__(delimiter)

        self.logger = None
        self.csvlogger = None
        if filename is not None:
            today = current_date()
            self.logfilename = f'{today}_{filename}'
            self.logger = self._create_logger(self.logfilename, work_dir)
            if csv:
                self.csvlogger = self._create_csv_logger(self.logfilename, work_dir)

    def log_every(self, iterable, print_freq, header=None):
        ''' Override intial method '''
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        flag = 1
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    printed_msg = log_msg.format(
                        i+1, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB)

                    dict_msg = dict(
                        date= get_date_time()[0], time= get_date_time()[1],
                        header=header, iter=i+1, total_iters=len(iterable),
                        eta=eta_string, iter_time=iter_time, data_time=data_time,
                        max_mem=torch.cuda.max_memory_allocated() / MB,
                        **self.meters
                    )

                    print(printed_msg)

                else:
                    printed_msg = log_msg.format(
                        i+1, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time))

                    dict_msg = dict(
                        date= get_date_time()[0], time= get_date_time()[1],
                        header=header, iter=i+1, total_iters=len(iterable),
                        eta=eta_string, iter_time=iter_time, data_time=data_time,
                        **self.meters
                    )

                    print(printed_msg)
                
                # Logs
                if self.logger is not None:
                    self.logger.info(printed_msg)
                if self.csvlogger is not None:
                    if flag == 1:
                        # New header to add meters
                        new_header = [*self.csvlogger.csvheader, *self.meters.keys()]
                        self.csvlogger.update_header(new_header)
                        flag = 0

                    self.csvlogger.add(dict_msg)

            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        total_msg = '{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable))
        print(total_msg)

        # Logs
        if self.logger is not None:
            self.logger.info(total_msg)
    
    def _create_logger(self, filename, directory):
        ''' Method to create a logger object '''
        logpath = os.path.join(directory, filename + '.log')
        handler = logging.FileHandler(logpath, 'a+')
        formatter = logging.Formatter(fmt='%(asctime)s - %(message)s', datefmt='%d-%m-%Y %H:%M:%S')
        handler.setFormatter(formatter)

        logger = logging.getLogger(filename)
        logger.setLevel(logging.INFO)  
        logger.addHandler(handler)

        return logger
    
    def _create_csv_logger(self, filename, directory):
        ''' Method to save logs into a CSV file '''
        csvheader = ['date','time','header','iter','total_iters','eta','iter_time','data_time','max_mem']
        csvlogger = CSVLogger(filename, directory, csvheader)
        return csvlogger

class CSVLogger:
    ''' Class for creating object to hold and save logs to CSV '''
    def __init__(self, filename, directory, header):
        self.csvpath = os.path.join(directory, filename + '.csv')
        self.filename = filename
        self.directory = directory
        self.csvheader = header

        if os.path.exists(self.csvpath) is not True:
            self._create_file(filename, directory, header)
    
    def add(self, msg_dict):
        ''' Add a log to the CSV file '''
        with open(self.csvpath, 'a+', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames= self.csvheader)
            writer.writerow(msg_dict)
            csvfile.close()
    
    def update_header(self, new_header):
        ''' update header if no data '''
        self.csvheader = new_header

        with open(self.csvpath, 'r') as rfile:
            reader = csv.DictReader(rfile, fieldnames=self.csvheader)

            if len(list(reader)) < 2:

                with open(self.csvpath, 'w', newline='') as ofile:
                    writer = csv.DictWriter(ofile, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    ofile.close()

            rfile.close()

    def _create_file(self, filename, directory, header):
        with open(self.csvpath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            csvfile.close()