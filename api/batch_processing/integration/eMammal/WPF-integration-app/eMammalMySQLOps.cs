using System;
using System.Configuration;
using System.Text;
using System.Data;
using MySql.Data.MySqlClient;
using NLog;

namespace eMammal_integration_application
{
    public class eMammalMySQLOps
    {
        Logger logger = LogManager.GetCurrentClassLogger();

        eMammalIntegrationWindow window;

        private string mysqlConnectionstring = ConfigurationManager.AppSettings["mysqlConnectionstring"].ToString();
        MySqlConnection connection = new MySqlConnection();

        public eMammalMySQLOps(eMammalIntegrationWindow window)
        {
            this.window = window;
            connection = new MySqlConnection(mysqlConnectionstring);
        }
        public eMammalMySQLOps()
        {
            connection = new MySqlConnection(mysqlConnectionstring);
        }

        public bool OpenConnectionIfNotOpen(bool returnOnError = false)
        {
            try
            {
                if (connection.State == ConnectionState.Closed)
                {
                    logger.Info(Constants.LOG_OPEN_CLOSED_DATABASE_CONNECTION);

                    connection.Open();
                }
                logger.Info(Constants.LOG_OPENING_CLOSED_DATABASE_CONNECTION_SUCCESSFULL);

                return true;
            }
            catch (Exception ex)
            {
                if (returnOnError)
                    return false;

                throw;
            }
        }
        public void CloseConnection()
        {
            try
            {
                if (connection.State == ConnectionState.Open)
                {
                    connection.Close();
                    logger.Info(Constants.LOG_CLOSING_OPEN_DATABASE_CONNECTION);
                }
                else
                    logger.Info(Constants.LOG_DATABASE_CONNECTION_NOT_OPEN);
            }
            catch (Exception ex)
            {
                logger.Info(Constants.LOG_ERROR_WHILE_CLOSING_DATABASE_CONNECTION);
                logger.Error(ex.ToString());
            }
        }

        public bool IsConnectionOpen()
        {
            try
            {
                if (connection.State == ConnectionState.Open)
                    return true;

                return false;
            }
            catch (Exception ex)
            {
                return false;
            }
        }
        // TODO: add error checking return null on error
        /// <summary>
        /// This function called for select statements, returning multiple rows
        /// </summary>
        /// <param name="query"></param>
        /// <returns></returns>
        public DataTable GeData(string query)
        {
            DataTable dt = new DataTable();
            using (MySqlCommand command = new MySqlCommand(query, connection))
            {
                command.CommandType = CommandType.Text;
                dt.Load(command.ExecuteReader());

            }
            return dt;
        }

        /// <summary>
        /// This function is called for inserting or updating data in the DB
        /// </summary>
        /// <param name="query">SQL query string</param>
        public void ExecuteQuery(string query)
        {
            //using (MySqlConnection connection = new MySqlConnection(mysqlConnectionstring))
            //{
            //connection.Open();
            using (MySqlCommand command = new MySqlCommand(query, connection))
            {
                command.CommandType = CommandType.Text;
                command.CommandText = query;

                int result = command.ExecuteNonQuery();
            }
        }

        /// <summary>
        /// This function is called for returning a single value from DB
        /// </summary>
        /// <param name="query">SQL query string</param>
        /// <returns></returns>
        public object ExecuteScalar(string query)
        {
            OpenConnectionIfNotOpen();

            using (MySqlCommand command = new MySqlCommand(query, connection))
            {
                command.CommandType = CommandType.Text;
                command.CommandText = query;
                Object result = null;

                result = command.ExecuteScalar();

                return result;
            }
        }
        /// <summary>
        /// Add unique for sequenceid, projecttaxaid to prevent duplicate inserts
        /// </summary>
        public void AddUniqueKeySequenceTaxa()
        {
            string sql = " SELECT constraint_name" +
                         " FROM information_schema.TABLE_CONSTRAINTS" +
                         " WHERE table_name = 'emammal_sequence_annotation'" +
                         " AND constraint_name = 'ai4e_unique_key'";

            logger.Info(Constants.LOG_CHECKING_IF_UNIQUE_KEY_ALREADY_EXISTS);

            var result = ExecuteScalar(sql);
            if (result == null)
            {
                sql = " ALTER TABLE emammal_sequence_annotation " +
                      " ADD CONSTRAINT ai4e_unique_key UNIQUE KEY(sequence_id, project_taxa_id); ";

                logger.Info(Constants.LOG_ADDING_UNIQUE_KEY_CONSTRAINT);
                logger.Info(sql);

                ExecuteQuery(sql);
            }
            else
            {
                logger.Info(Constants.LOG_UNIQUE_KEY_ALREADY_EXISTS);
            }
        }


        /// <summary>
        /// Get sequenceids for all the images in a deployment
        /// </summary>
        /// <param name="deploymentId"></param>
        /// <returns></returns>
        public DataTable GetsequenceIDsfromDB(int deploymentId)
        {
            string sql = string.Format(" SELECT b.raw_name, b.image_sequence_id " +
                                       " FROM wild_ID.image_sequence a, wild_id.image b " +
                                       " WHERE a.image_sequence_id = b.image_sequence_id " +
                                       " AND a.deployment_id = {0}; ", deploymentId);

            string mysqlConnectionstring = ConfigurationManager.AppSettings["mysqlConnectionstring"].ToString();
            DataTable dt = new DataTable("imageSequences");

            using (MySqlConnection connection = new MySqlConnection(mysqlConnectionstring))
            {
                OpenConnectionIfNotOpen();

                using (MySqlDataAdapter adapter = new MySqlDataAdapter(sql, connection))
                {
                    adapter.Fill(dt);
                    return dt;
                }
            }
        }

        public DataTable GetEmammalTaxas(int projectId)
        {
            string sql = string.Format(" SELECT species, emammal_project_taxa_id FROM wild_id.emammal_project_taxa " +
                                       " WHERE project_id = {0}", projectId);

            DataTable dt = GetDataTable(sql, "ProjectDetails");
            return dt;

        }

        public DataTable GetProjectDetails()
        {
            // Get eMammal project name and ids
            string sql = " SELECT e.project_id, " +
                         " CONCAT('p', '-', e.project_id, ' ', p.name ) as name " +
                         " FROM wild_id.project p, wild_id.emammal_project e " +
                         " WHERE p.project_id = e.project_id ";

            DataTable dt = GetDataTable(sql, "ProjectDetails");
            return dt;
        }
        public DataTable GetSubProjectDetails(string projectId)
        {
            // Get eMammal project name and ids
            string sql = string.Format(" SELECT e.event_id, " +
                                       " CONCAT('sp', '-', e.event_id, ' ', e.name ) as name " +
                                       " FROM wild_id.event e " +
                                       " WHERE e.project_id = {0} ", projectId);

            DataTable dt = GetDataTable(sql, "SubProjectDetails");

            return dt;
        }
        public DataTable GetDeploymentDetails(out bool success, string eventId)
        {
            success = false;

            // Get eMammal project name and ids
            string sql = string.Format(" SELECT d.deployment_id, " +
                                       " CONCAT('d', '-', d.deployment_id, ' ', d.name ) as name " +
                                       " FROM deployment d, emammal_deployment e " +
                                       " WHERE d.deployment_id = e.deployment_id " +
                                       " AND event_id = {0} ", eventId);


            DataTable dt = GetDataTable(sql, "DeploymentsDetails");
            return dt;
        }

        public DataTable GetDataTable(string sql, string type)
        {
            DataTable dt = new DataTable();

            OpenConnectionIfNotOpen();

            using (MySqlDataAdapter adapter = new MySqlDataAdapter(sql, connection))
            {
                adapter.Fill(dt);
                return dt;
            }
        }

        public StringBuilder GetBulkInsertInitialString()
        {
            StringBuilder sql = new StringBuilder("INSERT INTO wild_id.emammal_sequence_annotation(sequence_id, project_taxa_id, total_count) VALUES ");
            return sql;
        }

        public bool BulkInsertAnnotations(StringBuilder sql)
        {
            string loginfo = "";

            string sqlString = sql.ToString().Remove(sql.Length - 1);

            sqlString += " ON DUPLICATE KEY UPDATE " +
                         " sequence_id = VALUES(sequence_id)," +
                         " project_taxa_id = VALUES(project_taxa_id), " +
                         " total_count = VALUES(total_count);";

            loginfo += "\n" + sqlString;

            OpenConnectionIfNotOpen();


            using (MySqlCommand cmd = new MySqlCommand(sql.ToString(), connection))
            {
                cmd.CommandType = CommandType.Text;
                cmd.CommandText = sqlString;
                cmd.ExecuteNonQuery();
            }
            return true;
        }

        public DataTable GetImagesForDeployment(int deploymentId)
        {

            logger.Info("Starting verification...");
            string sql = string.Format(" SELECT b.raw_name, b.image_sequence_id, deployment_id, d.common_name " +
                                       " FROM wild_ID.image_sequence a, wild_id.image b, " +
                                       " wild_id.emammal_sequence_annotation c, " +
                                       " wild_id.emammal_project_taxa d " +
                                       " WHERE a.image_sequence_id = b.image_sequence_id " +
                                       " AND c.sequence_id = a.image_sequence_id " +
                                       " AND c.project_taxa_id = d.emammal_project_taxa_id " +
                                       " AND a.deployment_id = {0} order by b.raw_name", deploymentId);


            logger.Info(sql);

            DataTable dt = new DataTable();
            dt = GetDataTable(sql, "");

            return dt;

        }

    }

}
