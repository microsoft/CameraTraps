using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.RightsManagement;
using System.Text;
using System.Threading.Tasks;

namespace eMammal_integration_application
{
    static class Constants
    {
        // Category constants
        public const string animal = "1";
        public const string person = "2";
        public const string vehicle = "3";

        // Message constants
        public const string DATABASE_CONNECTION_ERROR = "Cannot connect to the eMammal database. Please ensure that you have opened the eMammal app and you have logged into the app. " +
                                                        "Once you opened the eMammal application, this application will automatically refresh.";
        //public const string DATABASE_CONNECTION_ERROR = "Cannot connect to the eMammal database. Please ensure that you have opened the eMammal app and you have logged into the app. ";
                                                      

        public const string NO_JSON_FILE_ERROR = "Please select a JSON detections file";
        public const string DATABASE_AVAILABLE = "Application is now able to connect to the eMammal database";

        //log messages
        public const string LOG_MESSAGE_APP_CONNECTED_TO_DATABASE = "App successfully connected to the eMammal database";
        public const string LOG_MESSAGE_PROJECT_LOADED = "Projects loaded";
        public const string LOG_APP_COULD_NOT_CONNECT_TO_DATABASE = "App could not connect to the eMammal database";
        public const string LOG_APP_CLOSING = "App Closing";
        public const string LOG_CLOSING_OPEN_DATABASE_CONNECTION = "Closing open database connection";
        public const string LOG_DATABASE_CONNECTION_NOT_OPEN = "Database connection not open";
        public const string LOG_ERROR_WHILE_CLOSING_DATABASE_CONNECTION = "Error occurred while trying to close database connection";
        public const string LOG_OPEN_CLOSED_DATABASE_CONNECTION = "Opening closed connection";
        public const string LOG_OPENING_CLOSED_DATABASE_CONNECTION_SUCCESSFULL = "Opening closed database connection was successfull";
        public const string LOG_ERROR_WHILE_OPENING_DATABASE_CONNECTION = "Error occurred while opening database connection";
        public const string LOG_ADDING_UNIQUE_KEY_CONSTRAINT = "Adding unique key constraint";
        public const string LOG_CHECKING_IF_UNIQUE_KEY_ALREADY_EXISTS = "Checking if unique key already exists in the database";
        public const string LOG_UNIQUE_KEY_ALREADY_EXISTS = "Unique key already exists in the database";
        public const string LOG_START_PROCESSING_IMAGES = "Starting image processing";
        public const string LOG_GETTING_IMAGE_SEQUENCE_DATA_FROM_DB = "Getting image sequence data from the database";
        public const string LOG_COULD_NOT_RETRIEVE_IMAGE_SEQUENCES_FROM_DATABASE = "Could not retrive image sequences from the database";
        public const string LOG_NUM_IMAGE_SEQUENCES = "Number of image sequences returned from DB: ";
        public const string LOG_ITERATING_IMAGES_IN_JSON_FILE = "Iterating through the images in the JSON file";

        //Progress messages
        public const string PROCESSING_IMAGES = "Processing images...";
        public const string PROGRESS_GETTING_IMAGE_SEQUENCE_DATA_FROM_DB = "Getting image sequence data from the database";
        public const string PROGRESS_UPDATING_ANNOTATIONS_IN_DB = "Updating annotations in the database";
        public const string PROGRESS_CONTINUING_WITH_NEXT_IMAGE = "Continuing with next image";

        //Log and Progress messages
        public const string INSERTING_DETECTIONS = "Inserting detections";
        public const string INSERTING_REMAINING_DETECTIONS = "Inserting remaining detections";
     
        public const string ANNOTATIONS_ADDED_FOR_ALL_IMAGES = "Annotations added for all images in eMammal database";

        //Error messages
        public const string ERROR_WHILE_VERIFYING_ANNOTATIONS_IN_DB = "Error occurred while verifying annotations in eMammal database";




    }
}
