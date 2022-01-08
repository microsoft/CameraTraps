using System.Drawing;
using System.Windows.Forms;

namespace CameraTrapJsonManagerApp
{
    partial class Form
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Form));
            this.openFileDialog = new System.Windows.Forms.OpenFileDialog();
            this.backgroundWorkerMain = new System.ComponentModel.BackgroundWorker();
            this.backgroundWorkerProgressReporter = new System.ComponentModel.BackgroundWorker();
            this.buttonBrowse = new System.Windows.Forms.Button();
            this.textboxConfidenceThreshold = new System.Windows.Forms.TextBox();
            this.lblInputFile = new System.Windows.Forms.Label();
            this.lblConfidenceThreshold = new System.Windows.Forms.Label();
            this.buttonSubmit = new System.Windows.Forms.Button();
            this.textboxQuery = new System.Windows.Forms.TextBox();
            this.labelQuery = new System.Windows.Forms.Label();
            this.textboxReplacement = new System.Windows.Forms.TextBox();
            this.labelReplacement = new System.Windows.Forms.Label();
            this.comboBoxSplitFolderMode = new System.Windows.Forms.ComboBox();
            this.labelSplitFolderMode = new System.Windows.Forms.Label();
            this.textboxOutputFolderFile = new System.Windows.Forms.TextBox();
            this.lblOutputFolderFileName = new System.Windows.Forms.Label();
            this.textboxInputFile = new System.Windows.Forms.TextBox();
            this.panelMain = new System.Windows.Forms.Panel();
            this.checkBoxEnableReplacement = new System.Windows.Forms.CheckBox();
            this.buttonBrowseFolder = new System.Windows.Forms.Button();
            this.buttonHelp = new System.Windows.Forms.Button();
            this.panel1 = new System.Windows.Forms.Panel();
            this.checkBoxCopyJsonstoFolders = new System.Windows.Forms.CheckBox();
            this.checkBoxCreateFolders = new System.Windows.Forms.CheckBox();
            this.checkBoxMakeFolderRelative = new System.Windows.Forms.CheckBox();
            this.checkBoxOverwriteJsonFiles = new System.Windows.Forms.CheckBox();
            this.labelConfidenceThresholdHelpInfo = new System.Windows.Forms.Label();
            this.panelSplitFolderMode = new System.Windows.Forms.Panel();
            this.textBoxSplitParameter = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.checkBoxSplitFolders = new System.Windows.Forms.CheckBox();
            this.statusGroupBox = new System.Windows.Forms.Panel();
            this.progressBar1 = new System.Windows.Forms.ProgressBar();
            this.labelProgressPercentage = new System.Windows.Forms.Label();
            this.richTextboxStatus = new System.Windows.Forms.RichTextBox();
            this.labelProgressMsg = new System.Windows.Forms.Label();
            this.folderBrowserDialog = new System.Windows.Forms.FolderBrowserDialog();
            this.panelMain.SuspendLayout();
            this.panel1.SuspendLayout();
            this.panelSplitFolderMode.SuspendLayout();
            this.statusGroupBox.SuspendLayout();
            this.SuspendLayout();
            // 
            // backgroundWorkerMain
            // 
            this.backgroundWorkerMain.WorkerReportsProgress = true;
            this.backgroundWorkerMain.DoWork += new System.ComponentModel.DoWorkEventHandler(this.backgroundWorker1_DoWork);
            this.backgroundWorkerMain.RunWorkerCompleted += new System.ComponentModel.RunWorkerCompletedEventHandler(this.backgroundWorkerMain_RunWorkerCompleted);
            // 
            // backgroundWorkerProgressReporter
            // 
            this.backgroundWorkerProgressReporter.WorkerReportsProgress = true;
            this.backgroundWorkerProgressReporter.ProgressChanged += new System.ComponentModel.ProgressChangedEventHandler(this.backgroundWorker2_ProgressChanged);
            // 
            // buttonBrowse
            // 
            this.buttonBrowse.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            this.buttonBrowse.Font = new System.Drawing.Font("Microsoft Sans Serif", 10.125F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.buttonBrowse.ForeColor = System.Drawing.Color.White;
            this.buttonBrowse.Location = new System.Drawing.Point(615, 49);
            this.buttonBrowse.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.buttonBrowse.Name = "buttonBrowse";
            this.buttonBrowse.Size = new System.Drawing.Size(86, 34);
            this.buttonBrowse.TabIndex = 68;
            this.buttonBrowse.Text = "Browse";
            this.buttonBrowse.UseVisualStyleBackColor = false;
            this.buttonBrowse.Click += new System.EventHandler(this.buttonBrowse_Click);
            // 
            // textboxConfidenceThreshold
            // 
            this.textboxConfidenceThreshold.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textboxConfidenceThreshold.ForeColor = System.Drawing.Color.Black;
            this.textboxConfidenceThreshold.Location = new System.Drawing.Point(194, 256);
            this.textboxConfidenceThreshold.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.textboxConfidenceThreshold.Name = "textboxConfidenceThreshold";
            this.textboxConfidenceThreshold.Size = new System.Drawing.Size(78, 27);
            this.textboxConfidenceThreshold.TabIndex = 69;
            this.textboxConfidenceThreshold.TextChanged += new System.EventHandler(this.textboxConfidenceThreshold_TextChanged);
            // 
            // lblInputFile
            // 
            this.lblInputFile.AutoSize = true;
            this.lblInputFile.BackColor = System.Drawing.Color.Transparent;
            this.lblInputFile.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblInputFile.ForeColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.lblInputFile.Location = new System.Drawing.Point(4, 56);
            this.lblInputFile.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.lblInputFile.Name = "lblInputFile";
            this.lblInputFile.Size = new System.Drawing.Size(75, 19);
            this.lblInputFile.TabIndex = 70;
            this.lblInputFile.Text = "Input file:";
            // 
            // lblConfidenceThreshold
            // 
            this.lblConfidenceThreshold.AutoSize = true;
            this.lblConfidenceThreshold.BackColor = System.Drawing.Color.Transparent;
            this.lblConfidenceThreshold.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblConfidenceThreshold.ForeColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.lblConfidenceThreshold.Location = new System.Drawing.Point(4, 260);
            this.lblConfidenceThreshold.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.lblConfidenceThreshold.Name = "lblConfidenceThreshold";
            this.lblConfidenceThreshold.Size = new System.Drawing.Size(159, 19);
            this.lblConfidenceThreshold.TabIndex = 71;
            this.lblConfidenceThreshold.Text = "Confidence threshold:";
            // 
            // buttonSubmit
            // 
            this.buttonSubmit.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            this.buttonSubmit.FlatAppearance.BorderColor = System.Drawing.Color.RoyalBlue;
            this.buttonSubmit.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.buttonSubmit.ForeColor = System.Drawing.Color.White;
            this.buttonSubmit.Location = new System.Drawing.Point(5, 452);
            this.buttonSubmit.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.buttonSubmit.Name = "buttonSubmit";
            this.buttonSubmit.Size = new System.Drawing.Size(492, 45);
            this.buttonSubmit.TabIndex = 72;
            this.buttonSubmit.Text = "Process";
            this.buttonSubmit.UseVisualStyleBackColor = false;
            this.buttonSubmit.Click += new System.EventHandler(this.buttonSubmit_Click);
            // 
            // textboxQuery
            // 
            this.textboxQuery.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textboxQuery.ForeColor = System.Drawing.Color.Black;
            this.textboxQuery.Location = new System.Drawing.Point(194, 150);
            this.textboxQuery.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.textboxQuery.Name = "textboxQuery";
            this.textboxQuery.Size = new System.Drawing.Size(194, 27);
            this.textboxQuery.TabIndex = 73;
            // 
            // labelQuery
            // 
            this.labelQuery.AutoSize = true;
            this.labelQuery.BackColor = System.Drawing.Color.Transparent;
            this.labelQuery.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelQuery.ForeColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.labelQuery.Location = new System.Drawing.Point(4, 156);
            this.labelQuery.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.labelQuery.Name = "labelQuery";
            this.labelQuery.Size = new System.Drawing.Size(55, 19);
            this.labelQuery.TabIndex = 74;
            this.labelQuery.Text = "Query:";
            // 
            // textboxReplacement
            // 
            this.textboxReplacement.Enabled = false;
            this.textboxReplacement.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textboxReplacement.ForeColor = System.Drawing.Color.Black;
            this.textboxReplacement.Location = new System.Drawing.Point(194, 192);
            this.textboxReplacement.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.textboxReplacement.Name = "textboxReplacement";
            this.textboxReplacement.Size = new System.Drawing.Size(194, 27);
            this.textboxReplacement.TabIndex = 75;
            // 
            // labelReplacement
            // 
            this.labelReplacement.AutoSize = true;
            this.labelReplacement.BackColor = System.Drawing.Color.Transparent;
            this.labelReplacement.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelReplacement.ForeColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.labelReplacement.Location = new System.Drawing.Point(4, 192);
            this.labelReplacement.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.labelReplacement.Name = "labelReplacement";
            this.labelReplacement.Size = new System.Drawing.Size(102, 19);
            this.labelReplacement.TabIndex = 76;
            this.labelReplacement.Text = "Replacement:";
            // 
            // comboBoxSplitFolderMode
            // 
            this.comboBoxSplitFolderMode.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBoxSplitFolderMode.FlatStyle = System.Windows.Forms.FlatStyle.Popup;
            this.comboBoxSplitFolderMode.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.comboBoxSplitFolderMode.ForeColor = System.Drawing.Color.Black;
            this.comboBoxSplitFolderMode.FormattingEnabled = true;
            this.comboBoxSplitFolderMode.Items.AddRange(new object[] {
            "Top",
            "Bottom",
            "NFromBottom",
            "NFromTop"});
            this.comboBoxSplitFolderMode.Location = new System.Drawing.Point(280, 12);
            this.comboBoxSplitFolderMode.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.comboBoxSplitFolderMode.Name = "comboBoxSplitFolderMode";
            this.comboBoxSplitFolderMode.Size = new System.Drawing.Size(132, 27);
            this.comboBoxSplitFolderMode.TabIndex = 77;
            // 
            // labelSplitFolderMode
            // 
            this.labelSplitFolderMode.AutoSize = true;
            this.labelSplitFolderMode.BackColor = System.Drawing.Color.Transparent;
            this.labelSplitFolderMode.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelSplitFolderMode.ForeColor = System.Drawing.Color.Black;
            this.labelSplitFolderMode.Location = new System.Drawing.Point(132, 15);
            this.labelSplitFolderMode.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.labelSplitFolderMode.Name = "labelSplitFolderMode";
            this.labelSplitFolderMode.Size = new System.Drawing.Size(123, 19);
            this.labelSplitFolderMode.TabIndex = 78;
            this.labelSplitFolderMode.Text = "Split folder mode:";
            // 
            // textboxOutputFolderFile
            // 
            this.textboxOutputFolderFile.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textboxOutputFolderFile.ForeColor = System.Drawing.Color.Black;
            this.textboxOutputFolderFile.Location = new System.Drawing.Point(194, 105);
            this.textboxOutputFolderFile.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.textboxOutputFolderFile.Name = "textboxOutputFolderFile";
            this.textboxOutputFolderFile.Size = new System.Drawing.Size(408, 27);
            this.textboxOutputFolderFile.TabIndex = 83;
            this.textboxOutputFolderFile.TextChanged += new System.EventHandler(this.textboxOutputFolderFile_TextChanged);
            // 
            // lblOutputFolderFileName
            // 
            this.lblOutputFolderFileName.AutoSize = true;
            this.lblOutputFolderFileName.BackColor = System.Drawing.Color.Transparent;
            this.lblOutputFolderFileName.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblOutputFolderFileName.ForeColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.lblOutputFolderFileName.Location = new System.Drawing.Point(4, 93);
            this.lblOutputFolderFileName.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.lblOutputFolderFileName.Name = "lblOutputFolderFileName";
            this.lblOutputFolderFileName.Size = new System.Drawing.Size(182, 38);
            this.lblOutputFolderFileName.TabIndex = 84;
            this.lblOutputFolderFileName.Text = "Enter output file / \nbrowse to select a folder:";
            // 
            // textboxInputFile
            // 
            this.textboxInputFile.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textboxInputFile.ForeColor = System.Drawing.Color.Black;
            this.textboxInputFile.Location = new System.Drawing.Point(194, 53);
            this.textboxInputFile.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.textboxInputFile.Name = "textboxInputFile";
            this.textboxInputFile.Size = new System.Drawing.Size(408, 27);
            this.textboxInputFile.TabIndex = 85;
            this.textboxInputFile.TextChanged += new System.EventHandler(this.textboxInputFile_TextChanged);
            // 
            // panelMain
            // 
            this.panelMain.BackColor = System.Drawing.Color.Transparent;
            this.panelMain.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.panelMain.Controls.Add(this.checkBoxEnableReplacement);
            this.panelMain.Controls.Add(this.buttonBrowseFolder);
            this.panelMain.Controls.Add(this.buttonHelp);
            this.panelMain.Controls.Add(this.panel1);
            this.panelMain.Controls.Add(this.labelConfidenceThresholdHelpInfo);
            this.panelMain.Controls.Add(this.panelSplitFolderMode);
            this.panelMain.Controls.Add(this.textboxInputFile);
            this.panelMain.Controls.Add(this.lblOutputFolderFileName);
            this.panelMain.Controls.Add(this.textboxOutputFolderFile);
            this.panelMain.Controls.Add(this.labelReplacement);
            this.panelMain.Controls.Add(this.textboxReplacement);
            this.panelMain.Controls.Add(this.labelQuery);
            this.panelMain.Controls.Add(this.textboxQuery);
            this.panelMain.Controls.Add(this.buttonSubmit);
            this.panelMain.Controls.Add(this.lblConfidenceThreshold);
            this.panelMain.Controls.Add(this.lblInputFile);
            this.panelMain.Controls.Add(this.textboxConfidenceThreshold);
            this.panelMain.Controls.Add(this.buttonBrowse);
            this.panelMain.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.panelMain.Location = new System.Drawing.Point(30, 14);
            this.panelMain.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.panelMain.Name = "panelMain";
            this.panelMain.Size = new System.Drawing.Size(775, 571);
            this.panelMain.TabIndex = 68;
            // 
            // checkBoxEnableReplacement
            // 
            this.checkBoxEnableReplacement.AutoSize = true;
            this.checkBoxEnableReplacement.BackColor = System.Drawing.Color.Transparent;
            this.checkBoxEnableReplacement.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxEnableReplacement.ForeColor = System.Drawing.Color.Black;
            this.checkBoxEnableReplacement.Location = new System.Drawing.Point(392, 185);
            this.checkBoxEnableReplacement.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.checkBoxEnableReplacement.Name = "checkBoxEnableReplacement";
            this.checkBoxEnableReplacement.Padding = new System.Windows.Forms.Padding(10, 11, 6, 6);
            this.checkBoxEnableReplacement.Size = new System.Drawing.Size(173, 40);
            this.checkBoxEnableReplacement.TabIndex = 100;
            this.checkBoxEnableReplacement.Text = "Enable replacement";
            this.checkBoxEnableReplacement.UseVisualStyleBackColor = false;
            this.checkBoxEnableReplacement.CheckedChanged += new System.EventHandler(this.enableReplacement_checkedChanged);
            // 
            // buttonBrowseFolder
            // 
            this.buttonBrowseFolder.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            this.buttonBrowseFolder.Font = new System.Drawing.Font("Microsoft Sans Serif", 10.125F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.buttonBrowseFolder.ForeColor = System.Drawing.Color.White;
            this.buttonBrowseFolder.Location = new System.Drawing.Point(614, 101);
            this.buttonBrowseFolder.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.buttonBrowseFolder.Name = "buttonBrowseFolder";
            this.buttonBrowseFolder.Size = new System.Drawing.Size(86, 34);
            this.buttonBrowseFolder.TabIndex = 99;
            this.buttonBrowseFolder.Text = "Browse";
            this.buttonBrowseFolder.UseVisualStyleBackColor = false;
            this.buttonBrowseFolder.Click += new System.EventHandler(this.buttonBrowseFolder_Click);
            // 
            // buttonHelp
            // 
            this.buttonHelp.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            this.buttonHelp.FlatAppearance.BorderColor = System.Drawing.Color.RoyalBlue;
            this.buttonHelp.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.buttonHelp.ForeColor = System.Drawing.Color.White;
            this.buttonHelp.Location = new System.Drawing.Point(501, 452);
            this.buttonHelp.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.buttonHelp.Name = "buttonHelp";
            this.buttonHelp.Size = new System.Drawing.Size(169, 45);
            this.buttonHelp.TabIndex = 98;
            this.buttonHelp.Text = "Help";
            this.buttonHelp.UseVisualStyleBackColor = false;
            this.buttonHelp.Click += new System.EventHandler(this.ButtonHelp_Click);
            // 
            // panel1
            // 
            this.panel1.BackColor = System.Drawing.Color.LightSteelBlue;
            this.panel1.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel1.Controls.Add(this.checkBoxCopyJsonstoFolders);
            this.panel1.Controls.Add(this.checkBoxCreateFolders);
            this.panel1.Controls.Add(this.checkBoxMakeFolderRelative);
            this.panel1.Controls.Add(this.checkBoxOverwriteJsonFiles);
            this.panel1.Location = new System.Drawing.Point(4, 380);
            this.panel1.Margin = new System.Windows.Forms.Padding(2);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(662, 53);
            this.panel1.TabIndex = 97;
            // 
            // checkBoxCopyJsonstoFolders
            // 
            this.checkBoxCopyJsonstoFolders.AutoSize = true;
            this.checkBoxCopyJsonstoFolders.BackColor = System.Drawing.Color.Transparent;
            this.checkBoxCopyJsonstoFolders.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxCopyJsonstoFolders.ForeColor = System.Drawing.Color.Black;
            this.checkBoxCopyJsonstoFolders.Location = new System.Drawing.Point(0, 3);
            this.checkBoxCopyJsonstoFolders.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.checkBoxCopyJsonstoFolders.Name = "checkBoxCopyJsonstoFolders";
            this.checkBoxCopyJsonstoFolders.Padding = new System.Windows.Forms.Padding(10, 11, 10, 11);
            this.checkBoxCopyJsonstoFolders.Size = new System.Drawing.Size(184, 45);
            this.checkBoxCopyJsonstoFolders.TabIndex = 92;
            this.checkBoxCopyJsonstoFolders.Text = "Copy jsons to folders";
            this.checkBoxCopyJsonstoFolders.UseVisualStyleBackColor = false;
            this.checkBoxCopyJsonstoFolders.CheckedChanged += new System.EventHandler(this.checkBoxCopyJsonsToFolders_CheckedChanged);
            // 
            // checkBoxCreateFolders
            // 
            this.checkBoxCreateFolders.AutoSize = true;
            this.checkBoxCreateFolders.BackColor = System.Drawing.Color.Transparent;
            this.checkBoxCreateFolders.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxCreateFolders.ForeColor = System.Drawing.Color.Black;
            this.checkBoxCreateFolders.Location = new System.Drawing.Point(178, 3);
            this.checkBoxCreateFolders.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.checkBoxCreateFolders.Name = "checkBoxCreateFolders";
            this.checkBoxCreateFolders.Padding = new System.Windows.Forms.Padding(10, 11, 6, 6);
            this.checkBoxCreateFolders.Size = new System.Drawing.Size(136, 40);
            this.checkBoxCreateFolders.TabIndex = 93;
            this.checkBoxCreateFolders.Text = "Create folders";
            this.checkBoxCreateFolders.UseVisualStyleBackColor = false;
            // 
            // checkBoxMakeFolderRelative
            // 
            this.checkBoxMakeFolderRelative.AutoSize = true;
            this.checkBoxMakeFolderRelative.BackColor = System.Drawing.Color.Transparent;
            this.checkBoxMakeFolderRelative.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxMakeFolderRelative.ForeColor = System.Drawing.Color.Black;
            this.checkBoxMakeFolderRelative.Location = new System.Drawing.Point(490, 3);
            this.checkBoxMakeFolderRelative.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.checkBoxMakeFolderRelative.Name = "checkBoxMakeFolderRelative";
            this.checkBoxMakeFolderRelative.Padding = new System.Windows.Forms.Padding(10, 11, 6, 6);
            this.checkBoxMakeFolderRelative.Size = new System.Drawing.Size(175, 40);
            this.checkBoxMakeFolderRelative.TabIndex = 90;
            this.checkBoxMakeFolderRelative.Text = "Make folder-relative";
            this.checkBoxMakeFolderRelative.UseVisualStyleBackColor = false;
            // 
            // checkBoxOverwriteJsonFiles
            // 
            this.checkBoxOverwriteJsonFiles.AutoSize = true;
            this.checkBoxOverwriteJsonFiles.BackColor = System.Drawing.Color.Transparent;
            this.checkBoxOverwriteJsonFiles.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxOverwriteJsonFiles.ForeColor = System.Drawing.Color.Black;
            this.checkBoxOverwriteJsonFiles.Location = new System.Drawing.Point(316, 3);
            this.checkBoxOverwriteJsonFiles.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.checkBoxOverwriteJsonFiles.Name = "checkBoxOverwriteJsonFiles";
            this.checkBoxOverwriteJsonFiles.Padding = new System.Windows.Forms.Padding(10, 11, 6, 6);
            this.checkBoxOverwriteJsonFiles.Size = new System.Drawing.Size(170, 40);
            this.checkBoxOverwriteJsonFiles.TabIndex = 91;
            this.checkBoxOverwriteJsonFiles.Text = "Overwrite json files";
            this.checkBoxOverwriteJsonFiles.UseVisualStyleBackColor = false;
            // 
            // labelConfidenceThresholdHelpInfo
            // 
            this.labelConfidenceThresholdHelpInfo.AutoSize = true;
            this.labelConfidenceThresholdHelpInfo.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelConfidenceThresholdHelpInfo.ForeColor = System.Drawing.Color.WhiteSmoke;
            this.labelConfidenceThresholdHelpInfo.Location = new System.Drawing.Point(278, 258);
            this.labelConfidenceThresholdHelpInfo.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.labelConfidenceThresholdHelpInfo.Name = "labelConfidenceThresholdHelpInfo";
            this.labelConfidenceThresholdHelpInfo.Size = new System.Drawing.Size(174, 19);
            this.labelConfidenceThresholdHelpInfo.TabIndex = 96;
            this.labelConfidenceThresholdHelpInfo.Text = "(value between 0 and 1)";
            // 
            // panelSplitFolderMode
            // 
            this.panelSplitFolderMode.BackColor = System.Drawing.Color.LightSteelBlue;
            this.panelSplitFolderMode.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelSplitFolderMode.Controls.Add(this.textBoxSplitParameter);
            this.panelSplitFolderMode.Controls.Add(this.label1);
            this.panelSplitFolderMode.Controls.Add(this.labelSplitFolderMode);
            this.panelSplitFolderMode.Controls.Add(this.checkBoxSplitFolders);
            this.panelSplitFolderMode.Controls.Add(this.comboBoxSplitFolderMode);
            this.panelSplitFolderMode.ForeColor = System.Drawing.Color.Black;
            this.panelSplitFolderMode.Location = new System.Drawing.Point(4, 316);
            this.panelSplitFolderMode.Margin = new System.Windows.Forms.Padding(2);
            this.panelSplitFolderMode.Name = "panelSplitFolderMode";
            this.panelSplitFolderMode.Size = new System.Drawing.Size(662, 53);
            this.panelSplitFolderMode.TabIndex = 95;
            this.panelSplitFolderMode.Paint += new System.Windows.Forms.PaintEventHandler(this.panelSplitFolderMode_Paint);
            // 
            // textBoxSplitParameter
            // 
            this.textBoxSplitParameter.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textBoxSplitParameter.ForeColor = System.Drawing.Color.Black;
            this.textBoxSplitParameter.Location = new System.Drawing.Point(561, 12);
            this.textBoxSplitParameter.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.textBoxSplitParameter.Name = "textBoxSplitParameter";
            this.textBoxSplitParameter.Size = new System.Drawing.Size(78, 27);
            this.textBoxSplitParameter.TabIndex = 91;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.BackColor = System.Drawing.Color.Transparent;
            this.label1.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.ForeColor = System.Drawing.Color.Black;
            this.label1.Location = new System.Drawing.Point(436, 15);
            this.label1.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(112, 19);
            this.label1.TabIndex = 90;
            this.label1.Text = "Split parameter:";
            // 
            // checkBoxSplitFolders
            // 
            this.checkBoxSplitFolders.AutoSize = true;
            this.checkBoxSplitFolders.BackColor = System.Drawing.Color.Transparent;
            this.checkBoxSplitFolders.Font = new System.Drawing.Font("Calibri", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.checkBoxSplitFolders.ForeColor = System.Drawing.Color.Black;
            this.checkBoxSplitFolders.Location = new System.Drawing.Point(2, 3);
            this.checkBoxSplitFolders.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.checkBoxSplitFolders.Name = "checkBoxSplitFolders";
            this.checkBoxSplitFolders.Padding = new System.Windows.Forms.Padding(10, 11, 6, 6);
            this.checkBoxSplitFolders.Size = new System.Drawing.Size(121, 40);
            this.checkBoxSplitFolders.TabIndex = 89;
            this.checkBoxSplitFolders.Text = "Split folders";
            this.checkBoxSplitFolders.UseVisualStyleBackColor = false;
            // 
            // statusGroupBox
            // 
            this.statusGroupBox.BackColor = System.Drawing.Color.White;
            this.statusGroupBox.Controls.Add(this.progressBar1);
            this.statusGroupBox.Controls.Add(this.labelProgressPercentage);
            this.statusGroupBox.Controls.Add(this.richTextboxStatus);
            this.statusGroupBox.Controls.Add(this.labelProgressMsg);
            this.statusGroupBox.Font = new System.Drawing.Font("Microsoft Sans Serif", 11F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.statusGroupBox.Location = new System.Drawing.Point(1, 535);
            this.statusGroupBox.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.statusGroupBox.Name = "statusGroupBox";
            this.statusGroupBox.Padding = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.statusGroupBox.Size = new System.Drawing.Size(842, 232);
            this.statusGroupBox.TabIndex = 73;
            // 
            // progressBar1
            // 
            this.progressBar1.Location = new System.Drawing.Point(2, 39);
            this.progressBar1.Margin = new System.Windows.Forms.Padding(2);
            this.progressBar1.Name = "progressBar1";
            this.progressBar1.Size = new System.Drawing.Size(840, 23);
            this.progressBar1.TabIndex = 73;
            // 
            // labelProgressPercentage
            // 
            this.labelProgressPercentage.AutoSize = true;
            this.labelProgressPercentage.ForeColor = System.Drawing.Color.Black;
            this.labelProgressPercentage.Location = new System.Drawing.Point(3, 12);
            this.labelProgressPercentage.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.labelProgressPercentage.Name = "labelProgressPercentage";
            this.labelProgressPercentage.Size = new System.Drawing.Size(182, 18);
            this.labelProgressPercentage.TabIndex = 72;
            this.labelProgressPercentage.Text = "[labelProgressPercentage]";
            // 
            // richTextboxStatus
            // 
            this.richTextboxStatus.BackColor = System.Drawing.Color.Black;
            this.richTextboxStatus.BorderStyle = System.Windows.Forms.BorderStyle.None;
            this.richTextboxStatus.ForeColor = System.Drawing.Color.White;
            this.richTextboxStatus.Location = new System.Drawing.Point(2, 64);
            this.richTextboxStatus.Margin = new System.Windows.Forms.Padding(0);
            this.richTextboxStatus.Name = "richTextboxStatus";
            this.richTextboxStatus.ReadOnly = true;
            this.richTextboxStatus.Size = new System.Drawing.Size(840, 178);
            this.richTextboxStatus.TabIndex = 71;
            this.richTextboxStatus.Text = "";
            // 
            // labelProgressMsg
            // 
            this.labelProgressMsg.AutoSize = true;
            this.labelProgressMsg.Font = new System.Drawing.Font("Calibri", 13.875F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.labelProgressMsg.ForeColor = System.Drawing.Color.MediumBlue;
            this.labelProgressMsg.Location = new System.Drawing.Point(4, 8);
            this.labelProgressMsg.Margin = new System.Windows.Forms.Padding(0);
            this.labelProgressMsg.Name = "labelProgressMsg";
            this.labelProgressMsg.Size = new System.Drawing.Size(159, 23);
            this.labelProgressMsg.TabIndex = 69;
            this.labelProgressMsg.Text = "[labelProgressMsg]";
            // 
            // Form
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.AutoSize = true;
            this.BackColor = System.Drawing.Color.White;
            this.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("$this.BackgroundImage")));
            this.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.ClientSize = new System.Drawing.Size(840, 766);
            this.Controls.Add(this.statusGroupBox);
            this.Controls.Add(this.panelMain);
            this.ForeColor = System.Drawing.SystemColors.ActiveCaptionText;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Margin = new System.Windows.Forms.Padding(2, 3, 2, 3);
            this.MaximizeBox = false;
            this.Name = "Form";
            this.StartPosition = System.Windows.Forms.FormStartPosition.Manual;
            this.Text = "Camera Trap API Output Manager";
            this.Load += new System.EventHandler(this.form_Load);
            this.panelMain.ResumeLayout(false);
            this.panelMain.PerformLayout();
            this.panel1.ResumeLayout(false);
            this.panel1.PerformLayout();
            this.panelSplitFolderMode.ResumeLayout(false);
            this.panelSplitFolderMode.PerformLayout();
            this.statusGroupBox.ResumeLayout(false);
            this.statusGroupBox.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion
        private System.Windows.Forms.OpenFileDialog openFileDialog;
        private System.ComponentModel.BackgroundWorker backgroundWorkerMain;
        private System.ComponentModel.BackgroundWorker backgroundWorkerProgressReporter;
        private Button buttonBrowse;
        private TextBox textboxConfidenceThreshold;
        private Label lblInputFile;
        private Label lblConfidenceThreshold;
        private Button buttonSubmit;
        private TextBox textboxQuery;
        private Label labelQuery;
        private TextBox textboxReplacement;
        private Label labelReplacement;
        private ComboBox comboBoxSplitFolderMode;
        private Label labelSplitFolderMode;
        private TextBox textboxOutputFolderFile;
        private Label lblOutputFolderFileName;
        private TextBox textboxInputFile;
        private Panel panelMain;
        private CheckBox checkBoxSplitFolders;
        private CheckBox checkBoxCopyJsonstoFolders;
        private CheckBox checkBoxCreateFolders;
        private CheckBox checkBoxOverwriteJsonFiles;
        private CheckBox checkBoxMakeFolderRelative;
        private Panel panelSplitFolderMode;
        private Label labelConfidenceThresholdHelpInfo;
        private Panel panel1;
        private Panel statusGroupBox;
        private Label labelProgressPercentage;
        private RichTextBox richTextboxStatus;
        private Label labelProgressMsg;
        private ProgressBar progressBar1;
        private TextBox textBoxSplitParameter;
        private Label label1;
        private Button buttonHelp;
        private FolderBrowserDialog folderBrowserDialog;
        private Button buttonBrowseFolder;
        private CheckBox checkBoxEnableReplacement;
    }
}