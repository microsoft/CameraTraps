import os
import logging
import time

from datetime import datetime
from pytz import timezone, utc
from logging.handlers import TimedRotatingFileHandler

log_path = "./log"

class Log(object):
    def __init__(self, name='log', level=logging.DEBUG):
        if not os.path.exists(log_path):
            os.mkdir(log_path)

        self.remove_old_logs()

        handler = TimedRotatingFileHandler("log/log.log", when='midnight', interval=1)
        handler.suffix = "%Y%m%d"

        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)

        self.logger = logging.getLogger(name)
        self.logger.addHandler(handler)
        self.logger.setLevel(level)
        logging.Formatter.converter = self.customTime


    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def customTime(self, *args):
        utc_dt = utc.localize(datetime.utcnow())
        my_tz = timezone("US/Pacific")
        converted = utc_dt.astimezone(my_tz)
        return converted.timetuple()

    def remove_old_logs(self):
        for f in os.listdir(log_path):
            file_path = log_path + "/"  + f
            now = time.time()
            if os.stat(file_path).st_mtime < now - 7 * 86400:
                os.remove(file_path)
