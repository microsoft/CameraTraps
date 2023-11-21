using MySql.Data.MySqlClient;
using Newtonsoft.Json;
using NLog;
using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Threading;

namespace eMammal_integration_application
{

    /// <summary>
    /// Interaction logic for eMammalIntegrationWindow.xaml
    /// </summary>
    public partial class eMammalIntegrationWindow : Window
    {
        Logger logger = LogManager.GetCurrentClassLogger();

        eMammalMySQLOps db;

        eMammalIntegration eMammalIntegration;

        //double tabTopOriginalMargin;
        //double originalHeight;

        public eMammalIntegrationWindow()
        {
            InitializeComponent();
            db = new eMammalMySQLOps(this);

            //tabTopOriginalMargin = Tab.Margin.Top;

            eMammalIntegration = new eMammalIntegration(this);
        }


        private void WindowInitialized(object sender, EventArgs e)
        {
            WindowStartupLocation = System.Windows.WindowStartupLocation.CenterScreen;
        }
        private void WindowLoaded(object sender, RoutedEventArgs e)
        {
            if (db.OpenConnectionIfNotOpen(true))
            {
                logger.Info(Constants.LOG_MESSAGE_APP_CONNECTED_TO_DATABASE);

                Loadproject();
                logger.Info(Constants.LOG_MESSAGE_PROJECT_LOADED);
            }
            else
            {
                logger.Info(Constants.LOG_APP_COULD_NOT_CONNECT_TO_DATABASE);

                Common.SetMessage(this, Constants.DATABASE_CONNECTION_ERROR, true, true);

                this.IsEnabled = false;

                DisableButton(ButtonNext);
                DisableButton(ButtonBack);
                DisableButton(ButtonBrowse);

                this.Activate();

                Thread thread = new Thread(() => Common.CheckConnection(this, true));
                thread.Start();
            }
        }

        private void WindowClosing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            logger.Info(Constants.LOG_APP_CLOSING);

            db.CloseConnection();
        }

        private void TabSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (Tab.SelectedIndex == 0 | Tab.SelectedIndex == 1)
            {
                TabResults.Visibility = Visibility.Hidden;
            }
            if (Tab.SelectedIndex == 0)
            {
                ButtonBack.Visibility = Visibility.Hidden;
            }
            if (Tab.SelectedIndex == 1)
            {
                ButtonBack.Visibility = Visibility.Visible;
            }
        }
        private void ButtonNextClick(object sender, RoutedEventArgs e)
        {
            try
            {
                //this.Tab.Margin = new Thickness(Tab.Margin.Left, tabTopOriginalMargin, Tab.Margin.Right, Tab.Margin.Bottom);
                TabResults.Visibility = Visibility.Hidden;

                ResetControlsAfterProcessing();

                TextBlockInfo.Text = "";
                TextBlockInfo.Visibility = Visibility.Hidden;

                if (Tab.SelectedIndex == 0)
                {
                    if (String.IsNullOrEmpty(TextBoxJsonFile.Text))
                    {
                        SetInvalidJsonError(Constants.NO_JSON_FILE_ERROR);
                        return;
                    }
                    else
                    {
                        if (!IsJsonFile())
                            return;
                    }
                    TabClassMapping.IsEnabled = true;

                    Tab.SelectedIndex = 1;

                    LoadCategoryMappings();
                    ButtonBack.Visibility = Visibility.Visible;

                }
                else
                {
                    TabClassMapping.IsEnabled = false;
                    CanvasClassMapping.IsEnabled = false;

                    TabDetails.IsEnabled = false;

                    ButtonBack.IsEnabled = false;
                    ButtonNext.IsEnabled = false;
                    ButtonBack.Foreground = new SolidColorBrush(Colors.Gray);
                    ButtonNext.Foreground = new SolidColorBrush(Colors.Gray);

                    // Invoking change in one element to Refresh UI with the above changes
                    ButtonBack.Dispatcher.Invoke(() => ButtonBack.Foreground = new SolidColorBrush(Colors.Gray), DispatcherPriority.Background);

                    var data = LoadJson(TextBoxJsonFile.Text);
                    int deploymentId = (int)comboBoxDeployment.SelectedValue;

                    int eMammalBlankCategory = (int)cmbProjectTaxaMappingBlank.SelectedValue;
                    int eMammalAnimalCategory = (int)cmbProjectTaxaMappingAnimal.SelectedValue;
                    int eMammalPersonCategory = (int)cmbProjectTaxaMappingPerson.SelectedValue;
                    int eMammalVehicleCategory = (int)cmbProjectTaxaMappingVehicle.SelectedValue;

                    if (ProgressbarUpdateProgress.Maximum == 0)
                        ProgressbarUpdateProgress.Maximum = 1;

                    // This makes inserts into the eMammal app much faster
                    db.AddUniqueKeySequenceTaxa();

                    logger.Info(Constants.LOG_START_PROCESSING_IMAGES);

                    Common.ShowProgress(this, Constants.PROCESSING_IMAGES, 1);
                    bool success = eMammalIntegration.ProcessDetections(data, deploymentId, comboBoxDeployment.Text, new Category()
                    {
                        blank = eMammalBlankCategory,
                        animal = eMammalAnimalCategory,
                        person = eMammalPersonCategory,
                        vehicle = eMammalVehicleCategory
                    });

                    if (success)
                    {
                        ButtonVerify.Visibility = Visibility.Visible;
                        //Tab.Margin = new Thickness(Tab.Margin.Left, Tab.Margin.Top + 50, Tab.Margin.Right, Tab.Margin.Bottom);
                        //Tab.Visibility = Visibility.Hidden;

                        TextBlockInfo.Text = "";
                        TextBlockInfo.Inlines.Add("Processed all images in the JSON file.");
                        TextBlockInfo.Inlines.Add(" Open and close the eMammal application, then in the eMammal application select ");

                        TextBlockInfo.Inlines.Add("project >");

                        Run run = new Run(comboBoxProject.Text);
                        run.FontWeight = FontWeights.Bold;
                        TextBlockInfo.Inlines.Add(run);

                        TextBlockInfo.Inlines.Add(" sub-project >");
                        run = new Run(comboBoxSubProject.Text);
                        run.FontWeight = FontWeights.Bold;
                        TextBlockInfo.Inlines.Add(run);

                        TextBlockInfo.Inlines.Add(" deployment > ");
                        run = new Run(comboBoxDeployment.Text);
                        run.FontWeight = FontWeights.Bold;
                        TextBlockInfo.Inlines.Add(run);

                        TextBlockInfo.Foreground = new SolidColorBrush(Colors.Blue);
                        TextBlockInfo.Visibility = Visibility.Visible;

                        ReactivateButton(ButtonNext);
                        ReactivateButton(ButtonBack);

                        DisableButton(ButtonNext);

                        //this.Activate();

                        ResetControlsAfterProcessing();
                        DisableButton(ButtonNext);
                    }
                    else
                    {
                        ResetControlsAfterProcessing();
                        DisableButton(ButtonNext);
                    }
                }
            }
            catch (Exception ex)
            {
                Common.HideProgress(this);
                HandleExceptions(ex);
            }
        }

        public void ReactivateButton(Button button)
        {
            button.IsEnabled = true;
            button.Foreground = new System.Windows.Media.SolidColorBrush((Color)ColorConverter.ConvertFromString("#005ce6"));
        }

        private void DisableButton(Button button)
        {
            button.IsEnabled = false;
            button.Foreground = new SolidColorBrush(Colors.Gray);
        }

        /// <summary>, ta
        /// Remove progress bar and message, re-enable back and next buttons after processing
        /// </summary>
        private void ResetControlsAfterProcessing()
        {
            LabelProgress.Dispatcher.Invoke(() => LabelProgress.Content = "", DispatcherPriority.Background);
            LabelProgress.Dispatcher.Invoke(() => LabelProgress.Visibility
             = Visibility.Hidden, DispatcherPriority.Background);

            ProgressbarUpdateProgress.Dispatcher.Invoke(() =>
            ProgressbarUpdateProgress.Value = 0, DispatcherPriority.Background);
            ProgressbarUpdateProgress.Dispatcher.Invoke(() =>
            ProgressbarUpdateProgress.Visibility = Visibility.Hidden, DispatcherPriority.Background);

            TabDetails.IsEnabled = true;
            TabClassMapping.IsEnabled = true;
            CanvasClassMapping.IsEnabled = true;

            ButtonBack.IsEnabled = true;
            ButtonNext.IsEnabled = true;
            ButtonBack.Foreground = new System.Windows.Media.SolidColorBrush((Color)ColorConverter.ConvertFromString("#005ce6"));
            ButtonNext.Foreground = new System.Windows.Media.SolidColorBrush((Color)ColorConverter.ConvertFromString("#005ce6"));
        }

        /// <summary>
        /// Browse button for selecting a json file
        /// When the button is clicked file dialog opens from which user can
        /// select a json file
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void ButtonBrowseJsonClick(object sender, RoutedEventArgs e)
        {
            // TODO: change this code copied from web
            Microsoft.Win32.OpenFileDialog openFileDlg = new Microsoft.Win32.OpenFileDialog();

            Nullable<bool> result = openFileDlg.ShowDialog();
            if (result == true)
                TextBoxJsonFile.Text = openFileDlg.FileName;
        }

        private void ComboBoxProjectSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (IsComboBoxLoaded(sender))
                LoadSubProject();
        }
        private void ComboBoxSubProjectSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (IsComboBoxLoaded(sender))
                LoadDeployment();
        }

        /// <summary>
        /// Switch back to details tab
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void ButtonBackClick(object sender, RoutedEventArgs e)
        {
            Tab.SelectedIndex = 0;
            Tab.Visibility = Visibility.Visible;
            ButtonNext.IsEnabled = true;
            ButtonNext.Foreground = new System.Windows.Media.SolidColorBrush((Color)ColorConverter.ConvertFromString("#005ce6"));
            ButtonBack.Visibility = Visibility.Hidden;
            ButtonVerify.Visibility = Visibility.Hidden;
            TabResults.Visibility = Visibility.Hidden;
        }

        /// <summary>
        /// Text box changed event for json file textbox
        /// Hide error message and change border of textbox from red to black
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void TextBoxJsonTextChanged(object sender, TextChangedEventArgs e)
        {
            LabelJsonFileError.Visibility = Visibility.Hidden;
            TextBoxJsonFile.BorderBrush = Brushes.Black;
        }

        /// <summary>
        /// Loads the eMammal project id and names
        /// </summary>
        public void Loadproject()
        {
            DataTable dt = db.GetProjectDetails();
            FillDrodownLists(comboBoxProject, dt, "name", "project_id");
        }

        /// <summary>
        /// Loads the eMammal sub project id and names
        /// </summary>
        private void LoadSubProject()
        {
            DataTable dt = db.GetSubProjectDetails(comboBoxProject.SelectedValue.ToString());
            FillDrodownLists(comboBoxSubProject, dt, "name", "event_id");
        }

        /// <summary>
        /// Loads the eMammal deployment id and names
        /// </summary>
        private void LoadDeployment()
        {
            bool success;
            DataTable dt = db.GetDeploymentDetails(out success, comboBoxSubProject.SelectedValue.ToString());
            FillDrodownLists(comboBoxDeployment, dt, "name", "deployment_id");
        }

        private void FillDrodownLists(ComboBox combobox, DataTable dt, string displayMemberPath,
            string SelectedValuePath)
        {
            combobox.ItemsSource = dt.DefaultView;
            combobox.DisplayMemberPath = displayMemberPath;
            combobox.SelectedValuePath = SelectedValuePath;
            combobox.SelectedIndex = 0;
        }

        private bool IsComboBoxLoaded(object sender)
        {
            var comboBox = (ComboBox)sender;
            if (!comboBox.IsLoaded)
                return false;
            return true;
        }

        /// <summary>
        /// Sets error message in a label
        /// </summary>
        /// <param name="message"></param>
        private void SetInvalidJsonError(string message)
        {
            TextBoxJsonFile.BorderBrush = Brushes.Red;

            LabelJsonFileError.Content = message;
            LabelJsonFileError.Visibility = Visibility.Visible;
        }

        /// <summary>
        /// Checks if a file provided is a JSON file
        /// </summary>
        /// <returns></returns>
        private bool IsJsonFile()
        {
            string ext = System.IO.Path.GetExtension(TextBoxJsonFile.Text);
            if (ext.ToLower() != ".json")
            {
                SetInvalidJsonError("Please select a valid JSON file");
                return false;
            }
            return true;
        }

        /// <summary>
        /// 
        /// </summary>
        private void LoadCategoryMappings()
        {
            if (!cmbProjectTaxaMappingAnimal.HasItems)
            {
                var taxas = db.GetEmammalTaxas((int)comboBoxProject.SelectedValue);

                FillDrodownLists(cmbProjectTaxaMappingAnimal, taxas, "species", "emammal_project_taxa_id");
                FillDrodownLists(cmbProjectTaxaMappingPerson, taxas, "species", "emammal_project_taxa_id");
                FillDrodownLists(cmbProjectTaxaMappingVehicle, taxas, "species", "emammal_project_taxa_id");
                FillDrodownLists(cmbProjectTaxaMappingBlank, taxas, "species", "emammal_project_taxa_id");

                // Set the initial category in the category mapping dropdown lists
                SetPossibleCategory(cmbProjectTaxaMappingAnimal, "unknown animal");
                SetPossibleCategory(cmbProjectTaxaMappingPerson, "homo sapiens");
                SetPossibleCategory(cmbProjectTaxaMappingVehicle, "vehicle");
                SetPossibleCategory(cmbProjectTaxaMappingBlank, "no animal");
            }
        }
        ///<summary>
        /// Sets the initial category mapping in comboboxes 
        /// in the category mapping section
        /// </summary>
        /// <param name="comboBox"></param>
        /// <param name="text"></param>
        private void SetPossibleCategory(ComboBox comboBox, string text)
        {
            foreach (Object item in comboBox.Items)
            {
                DataRowView row = item as DataRowView;
                if (row != null)
                {
                    string displayValue = row["species"].ToString();
                    if (displayValue.ToLower() == text)
                        comboBox.SelectedIndex = comboBox.Items.IndexOf(item);
                }
            }
        }

        /// <summary>
        /// Loads json file into JsonData object
        /// </summary>
        /// <param name="inputFileName"></param>
        /// <returns></returns>
        private JsonData LoadJson(string inputFileName)
        {
            string json = File.ReadAllText(TextBoxJsonFile.Text);
            var data = JsonConvert.DeserializeObject<JsonData>(json);
            return data;
        }

        private void DisableTabs()
        {
            TabClassMapping.IsEnabled = false;
            TabDetails.IsEnabled = false;
        }
        private void EnableTabs()
        {
            TabClassMapping.IsEnabled = true;
            TabDetails.IsEnabled = true;
        }
        private void ButtonVerifyClick(object sender, RoutedEventArgs e)
        {
            try
            {
                logger.Info("Verifying images...");

                Mouse.OverrideCursor = System.Windows.Input.Cursors.Wait;

                //Common.delay(100);

                DisableButton(ButtonBack);
                DisableButton(ButtonNext);

                ButtonVerify.Visibility = Visibility.Hidden;

                TabDetails.IsEnabled = false;
                TabClassMapping.IsEnabled = false;
                TabResults.IsEnabled = true;

                int deploymentId = (int)comboBoxDeployment.SelectedValue;

                RichTextBoxResults.AppendText("\n");

                bool success = eMammalIntegration.VerifyAnnotations(deploymentId);

                Tab.SelectedIndex = 2;
                TabResults.Visibility = Visibility.Visible;
                TabResults.IsEnabled = true;

                ButtonVerify.Visibility = Visibility.Hidden;

                ResetControlsAfterProcessing();

                ReactivateButton(ButtonNext);
                ReactivateButton(ButtonBack);

                TabDetails.IsEnabled = true;
                TabClassMapping.IsEnabled = true;

                TextBlockInfo.Visibility = Visibility.Visible;

                if (!success)
                    TabResults.Visibility = Visibility.Hidden;
            }
            catch (Exception ex)
            {
                HandleExceptions(ex);
            }
            finally
            {
                Mouse.OverrideCursor = System.Windows.Input.Cursors.Arrow;
            }

        }

        private void HandleExceptions(Exception ex)
        {
            logger.Error(ex.ToString());

            if (ex is MySqlException)
            {
                HandleSQLExceptions(ex as MySqlException);
                Common.HideProgress(this);

                Thread thread = new Thread(() => Common.CheckConnection(this));
                thread.Start();

                return;
            }
            else
            {
                Common.HideProgress(this);
            }
            MessageBox.Show(ex.Message);
        }

        private void HandleSQLExceptions(MySqlException ex)
        {
            Tab.IsEnabled = false;
            Common.HideProgress(this);
            int number = -1;

            if (ex.InnerException != null && ex.InnerException is MySqlException)
            {
                number = ((MySqlException)ex.InnerException).Number;
                logger.Error(ex.InnerException.ToString());
            }

            if (number == 0 || ex.Number == 1042)
                Common.SetMessage(this, Constants.DATABASE_CONNECTION_ERROR, true, true);

            else if (ex.InnerException != null)
            {
                // no way to get the errorcode from inner exception therefore using this method
                if (ex.InnerException.InnerException != null)
                {
                    string errmsg = ex.InnerException.InnerException.Message;
                    if (errmsg.Contains("120.0.0.1:3307") && errmsg.Contains("No connection could be made"))
                        Common.SetMessage(this, Constants.DATABASE_CONNECTION_ERROR, true, true);
                }
                else
                    Common.SetMessage(this, ex.InnerException.Message, true, true);
            }
            else
                Common.SetMessage(this, ex.Message, true, true);
        }
    }

}
