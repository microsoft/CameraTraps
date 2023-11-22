using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.ComponentModel;
using System.Drawing;
using System.Windows.Forms;

namespace CameraTrapJsonManagerApp
{
    #region ProgressBar delegate declarations

    delegate void SetProgressBarStyleCallback(ProgressBarStyle style);
    delegate void SetProgressBarCallback(long value, long maximumvalue, 
        ProgressBarStyle barStyle, bool showProgressBar);
    delegate void SetStatusTextboxMsgCallback(string text);
    delegate void SetLabelProgressMsgOnlyCallback(Label lbl, string text);
    delegate void SetLabelProgressMsgandTextColorCallback(Label lbl, string text, Color textColor);
    delegate void ShowHideProgressCallback(bool show);

    #endregion
    public partial class Form : System.Windows.Forms.Form
    {
        SubsetJsonDetectorOutputOptions options = new SubsetJsonDetectorOutputOptions();

        public Form()
        {
            InitializeComponent();

            //center the form in the middle of the computer screen
            this.StartPosition = FormStartPosition.Manual;
            this.Left = (Screen.PrimaryScreen.Bounds.Width - this.Width) / 2;
        }

        #region Form and control events

        //private void Form_Paint(object sender, PaintEventArgs e)
        //{
        //    System.Drawing.Graphics graphics = e.Graphics;
        //    System.Drawing.Rectangle gradient_rectangle = new System.Drawing.Rectangle(0, 0, this.Width, this.Height);
        //    System.Drawing.Brush b = new System.Drawing.Drawing2D.LinearGradientBrush(gradient_rectangle, Color.White, Color.CornflowerBlue, 65f);
        //    graphics.FillRectangle(b, gradient_rectangle);
        //}

        private void panelSplitFolderMode_Paint(object sender, PaintEventArgs e)
        {
            panelSplitFolderMode.CreateGraphics().DrawRectangle(Pens.DimGray,
                comboBoxSplitFolderMode.Left - 1,
                comboBoxSplitFolderMode.Top - 1,
                comboBoxSplitFolderMode.Width + 1,
                comboBoxSplitFolderMode.Height + 1
             );
        }
        private void form_Load(object sender, EventArgs e)
        {
            comboBoxSplitFolderMode.SelectedIndex = 1;

            labelProgressMsg.Text = string.Empty;
            labelProgressPercentage.Text = string.Empty;

            textboxInputFile.Text = string.Empty;
            textboxOutputFolderFile.Text = string.Empty;
            this.Text += " (version " + System.Reflection.Assembly.GetExecutingAssembly().GetName().Version.ToString() + ")";

#if DEBUG            
            textboxInputFile.Text = @"g:\temp\test.json";
            textboxOutputFolderFile.Text = @"g:\temp\out";
            textboxConfidenceThreshold.Text = "";
            textBoxSplitParameter.Text = "1";
            comboBoxSplitFolderMode.SelectedItem = "NFromTop";
            checkBoxMakeFolderRelative.Checked = true;
            checkBoxOverwriteJsonFiles.Checked = true;
            checkBoxSplitFolders.Checked = true;            
#endif

        }
        private void buttonBrowse_Click(object sender, EventArgs e)
        {
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                textboxInputFile.Text = openFileDialog.FileName;
            }
        }
        private void buttonSubmit_Click(object sender, EventArgs e)
        {

            if (this.GetOptions())
            {
                labelProgressMsg.ForeColor = Color.MediumBlue;
                panelMain.Enabled = false;

                richTextboxStatus.Text = string.Empty;
                panelMain.Enabled = false;

                backgroundWorkerMain.RunWorkerAsync();

                labelProgressMsg.Focus();

                progressBar1.Visible = false;
            }
        }
        private void backgroundWorker1_DoWork(object sender, DoWorkEventArgs e)
        {
            try
            {
                backgroundWorkerProgressReporter.ReportProgress(0, new ProgressDetails { RemoveProgressInfo = true });

                SubsetJsonDetectorOutput sub = new SubsetJsonDetectorOutput(backgroundWorkerProgressReporter, options);

                var data = sub.SubsetJsonDetectorOutputMain(textboxInputFile.Text.Trim(), textboxOutputFolderFile.Text.Trim(), options, null);
                if (data != null)
                {
                    backgroundWorkerProgressReporter.ReportProgress(0, new ProgressDetails
                    {
                        ShowProgressBar = false,
                        SetlabelProgressMessage = true,
                        Message = "Program executed successfully"
                    });
                }
                else
                {
                    backgroundWorkerProgressReporter.ReportProgress(0, new ProgressDetails
                    {
                        ShowProgressBar = false,
                        SetlabelProgressMessage = true,
                        Message = "Program executed with error",
                        IsError = true
                    });
                }
                ShowHideProgressBar(false);
            }
            catch (Exception ex)
            {
                backgroundWorkerProgressReporter.ReportProgress(0, new ProgressDetails
                {
                    ShowProgressBar = false,
                    SetStatusTextBoxMessage = true,
                    Message = ex.ToString()
                });
            }
        }
        private void backgroundWorker2_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            ProgressDetails p = (ProgressDetails)e.UserState;

            if (p.RemoveProgressInfo)
            {
                SetLabelProgressMsg(labelProgressMsg, "");
                SetLabelProgressMsg(labelProgressPercentage, "");
                SetStatusTextboxMsg("");
                SetProgressBar(0);
            }

            if (p.ShowProgressBar)
            {
                if (p.style == ProgressBarStyle.Marquee)
                    SetProgressBar(10, 100, ProgressBarStyle.Marquee);
                else
                    SetProgressBar(p.CurrentCount, p.Maximum, ProgressBarStyle.Continuous);
            }
            if (p.SetlabelProgressMessage)
            {
                if (p.IsError)
                    SetLabelProgressMsgandTextColor(labelProgressMsg, p.Message, Color.Red);
                else
                    SetLabelProgressMsgandTextColor(labelProgressMsg, p.Message, Color.MediumBlue);

            }
            if(p.SetlabelProgressPercentageMessage)
            {
                if (p.IsError)
                    SetLabelProgressMsgandTextColor(labelProgressPercentage, p.Message, Color.Red);
                else
                    SetLabelProgressMsgandTextColor(labelProgressPercentage, p.Message, Color.Black);
            }
            if (p.SetStatusTextBoxMessage)
                SetStatusTextboxMsg(p.Message +"\n");
        }
        private void backgroundWorkerMain_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            panelMain.Enabled = true;
            richTextboxStatus.SelectionStart = richTextboxStatus.Text.Length;
            richTextboxStatus.ScrollToCaret();
            richTextboxStatus.Refresh();
        }
        private void textboxConfidenceThreshold_TextChanged(object sender, EventArgs e)
        {
            TextBoxChangeControlBorderColor(textboxConfidenceThreshold, Pens.Black);
        }

        private void checkBoxCopyJsonsToFolders_CheckedChanged(object sender, EventArgs e)
        {
            checkBoxCopyJsonstoFolders.ForeColor = Color.Black;
        }

        private void textboxInputFile_TextChanged(object sender, EventArgs e)
        {
            TextBoxChangeControlBorderColor(textboxInputFile, Pens.DimGray);
        }

        private void textboxOutputFolderFile_TextChanged(object sender, EventArgs e)
        {
            TextBoxChangeControlBorderColor(textboxOutputFolderFile, Pens.DimGray);
  
        }
        #endregion


        private bool GetOptions()
        {
            options.Query = textboxQuery.Text.Trim();

            if (checkBoxEnableReplacement.Checked)
                options.Replacement = textboxReplacement.Text; // .Trim();
            else
                options.Replacement = null;


            if (string.IsNullOrEmpty(textboxInputFile.Text.Trim()))
            {
                TextBoxChangeControlBorderColor(textboxInputFile, Pens.Red);
               
                backgroundWorkerProgressReporter.ReportProgress(0, new ProgressDetails
                {
                    ShowProgressBar = false,
                    SetlabelProgressMessage = true,
                    Message = "Please select an input file"
                });
                return false;
            }
            if (string.IsNullOrEmpty(textboxOutputFolderFile.Text.Trim()))
            {
                TextBoxChangeControlBorderColor(textboxOutputFolderFile, Pens.Red);

                backgroundWorkerProgressReporter.ReportProgress(0, new ProgressDetails
                {
                    ShowProgressBar = false,
                    SetlabelProgressMessage = true,
                    Message = "Please enter value for output file or folder name"
                });
                return false;
            }

            if (checkBoxCopyJsonstoFolders.Checked)
            {
                if (!(checkBoxSplitFolders.Checked && checkBoxMakeFolderRelative.Checked))
                {
                    checkBoxCopyJsonstoFolders.ForeColor = Color.Red;

                    backgroundWorkerProgressReporter.ReportProgress(0, new ProgressDetails
                    {
                        ShowProgressBar = false,
                        SetlabelProgressMessage = true,
                        Message = "'Copy jsons to folders' set without 'make folder-relative' and 'split folders'"
                    });

                    return false;
                }
            }

            if (!string.IsNullOrEmpty(textboxConfidenceThreshold.Text.Trim()))
            {
                double confidenceThreshold;
                
                if (double.TryParse(textboxConfidenceThreshold.Text.Trim(), out confidenceThreshold))
                    options.ConfidenceThreshold = confidenceThreshold;
                else
                {
                    panelMain.CreateGraphics().DrawRectangle(Pens.Red,
                                            textboxConfidenceThreshold.Left - 1,
                                            textboxConfidenceThreshold.Top - 1,
                                            textboxConfidenceThreshold.Width + 1,
                                            textboxConfidenceThreshold.Height + 1
                    );
                    
                    textboxConfidenceThreshold.Refresh();

                    backgroundWorkerProgressReporter.ReportProgress(0, new ProgressDetails
                    {
                        ShowProgressBar = false,
                        SetlabelProgressMessage = true,
                        Message = "Confidence threshold should be numeric"
                    });

                    return false;
                }

                if(confidenceThreshold < 0 | confidenceThreshold > 1)
                {
                    panelMain.CreateGraphics().DrawRectangle(Pens.Red,
                                           textboxConfidenceThreshold.Left - 1,
                                           textboxConfidenceThreshold.Top - 1,
                                           textboxConfidenceThreshold.Width + 1,
                                           textboxConfidenceThreshold.Height + 1
                   );

                    textboxConfidenceThreshold.Refresh();

                    backgroundWorkerProgressReporter.ReportProgress(0, new ProgressDetails
                    {
                        ShowProgressBar = false,
                        SetlabelProgressMessage = true,
                        Message = "Confidence threshold should be between 0 and 1"
                    });
                    return false;

                }
            }
            else
            {
                options.ConfidenceThreshold = -1;
            }

            options.SplitFolders = checkBoxSplitFolders.Checked;
            options.SplitFolderMode = comboBoxSplitFolderMode.SelectedItem.ToString();
            options.MakeFolderRelative = checkBoxMakeFolderRelative.Checked;
            options.OverwriteJsonFiles = checkBoxOverwriteJsonFiles.Checked;
            options.CopyJsonstoFolders = checkBoxCopyJsonstoFolders.Checked;
            options.CopyJsonstoFoldersDirectoriesMustExist = !(checkBoxCreateFolders.Checked);

            String splitParameterString = textBoxSplitParameter.Text.ToString().Trim();
            if (splitParameterString.Length > 0)
            {
                int splitParam;
                if (int.TryParse(splitParameterString, out splitParam))
                    options.nDirectoryParam = splitParam;
                else
                {
                    backgroundWorkerProgressReporter.ReportProgress(0, new ProgressDetails
                    {
                        ShowProgressBar = false,
                        SetlabelProgressMessage = true,
                        Message = "Split parameter should be an integer"
                    });

                    return false;
                }
            }

            return true;

        } // private bool GetOptions()

        private void TextBoxChangeControlBorderColor(TextBox txtbox, Pen penColor)
        {
            panelMain.CreateGraphics().DrawRectangle(penColor,
                                       txtbox.Left - 1,
                                       txtbox.Top - 1,
                                       txtbox.Width + 1,
                                       txtbox.Height + 1
             );

            txtbox.Refresh();
        }
    
        #region Progressbar delegates
   
        private void SetProgressBar(long value, long maximumvalue = 100,
            ProgressBarStyle barStyle = ProgressBarStyle.Blocks, bool showProgressBar = true)
        {
            if (this.progressBar1.InvokeRequired)
            {
                SetProgressBarCallback d = new SetProgressBarCallback(SetProgressBar);
                this.Invoke(d, new object[] { value, maximumvalue, barStyle, showProgressBar });
            }
            else
            {
                progressBar1.Visible = showProgressBar;
                if (showProgressBar)
                {
                    progressBar1.Style = barStyle;

                    if (barStyle == ProgressBarStyle.Marquee)
                    {
                        progressBar1.MarqueeAnimationSpeed = 30;
                        progressBar1.Value = 10;
                    }
                    else
                    {
                        progressBar1.Style = barStyle;
                        int percentage = SharedFunctions.GetProgressPercentage(value, maximumvalue);
                        //if (percentage < 1)
                        //{
                        //    percentage = 1;
                        //}
                        progressBar1.Maximum = 100;
                        //progressBar1.SetProgressNoAnimation(percentage);
                        progressBar1.Value = percentage;
                    }
                }
            }
        }
        private void SetProgressBarStyle(ProgressBarStyle style)
        {
            if (this.progressBar1.InvokeRequired)
            {
                SetProgressBarStyleCallback d = new SetProgressBarStyleCallback(SetProgressBarStyle);
                this.Invoke(d, new object[] { style });

            }
            else
            {
                if (style == ProgressBarStyle.Marquee)
                    progressBar1.Style = ProgressBarStyle.Marquee;
                else
                    progressBar1.Style = ProgressBarStyle.Blocks;
            }
        }
        private void ShowHideProgressBar(bool show)
        {
            if (this.progressBar1.InvokeRequired)
            {
                ShowHideProgressCallback d = new ShowHideProgressCallback(ShowHideProgressBar);
                this.Invoke(d, new object[] { show });

            }
            else
            {
                progressBar1.Visible = show;
            }
            if (this.labelProgressPercentage.InvokeRequired)
            {
                ShowHideProgressCallback d = new ShowHideProgressCallback(ShowHideProgressBar);
                this.Invoke(d, new object[] { show });

            }
            else
            {
                labelProgressPercentage.Visible = false;
            }
        }
        private void SetStatusTextboxMsg(string text)
        {
            // InvokeRequired required compares the thread ID of the
            // calling thread to the thread ID of the creating thread.
            // If these threads are different, it returns true.
            if (this.richTextboxStatus.InvokeRequired)
            {
                SetStatusTextboxMsgCallback d = new SetStatusTextboxMsgCallback(SetStatusTextboxMsg);
                this.Invoke(d, new object[] { text });
            }
            else
            {
                this.richTextboxStatus.AppendText(text);
            }
        }
        private void SetLabelProgressMsgandTextColor(Label lbl, string text, Color color)
        {
            // InvokeRequired required compares the thread ID of the
            // calling thread to the thread ID of the creating thread.
            // If these threads are different, it returns true.
            if (lbl.InvokeRequired)
            {
                SetLabelProgressMsgandTextColorCallback d = 
                    new SetLabelProgressMsgandTextColorCallback(SetLabelProgressMsgandTextColor);

                lbl.Invoke(d, new object[] { lbl, text, color });
            }
            else
            {
                lbl.Text = text;
                lbl.ForeColor = color;
            }          
        }
        private void SetLabelProgressMsg(Label lbl, string text)
        {
            // InvokeRequired required compares the thread ID of the
            // calling thread to the thread ID of the creating thread.
            // If these threads are different, it returns true.
            if (lbl.InvokeRequired)
            {
                SetLabelProgressMsgOnlyCallback d = 
                    new SetLabelProgressMsgOnlyCallback(SetLabelProgressMsg);
                lbl.Invoke(d, new object[] { lbl, text });
            }
            else
            {
                lbl.Text = text;
            }
        }

        #endregion

        private void ButtonHelp_Click(object sender, EventArgs e)
        {
            System.Diagnostics.Process.Start("https://github.com/ecologize/CameraTraps/blob/master/api/batch_processing/postprocessing/CameraTrapJsonManagerApp.md");
        }

        private void buttonBrowseFolder_Click(object sender, EventArgs e)
        {
            if (folderBrowserDialog.ShowDialog() == DialogResult.OK)
                textboxOutputFolderFile.Text = folderBrowserDialog.SelectedPath;
        }

        private void enableReplacement_checkedChanged(object sender, EventArgs e)
        {
            textboxReplacement.Enabled = checkBoxEnableReplacement.Checked;
        }
    }
}
