using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.IO;
using System.Threading;

namespace AnalyzeDV
{
   public class StartProcess1
    {
        int totalProcess = 1;//总任务数
        int maxParallelProcess = 4;//并行最大进程数
        int curRunningProcess = 0;//当前运行进程数
        public void Do()
        {
            DoEvents();
        }

        private System.Security.SecureString ToSecureString(string str){
            System.Security.SecureString result = new System.Security.SecureString();
            foreach(char cha in str){
                result.AppendChar(cha);
            }
            return result;
        }
        /// <summary>
        /// 执行进程
        /// </summary>
        private  async void DoEvents()
        {
            for (int i = 1; i <= totalProcess; i++)
            {
                try{
                    Console.WriteLine("Start " + i.ToString());
                    ProcessStartInfo processInfo = new ProcessStartInfo();
                
                    Console.WriteLine(File.Exists(@"/home/azureuser/Demo.exe"));
                    processInfo.FileName = @"/home/azureuser/Demo.exe";
                    processInfo.Arguments = i.ToString()+' ' + totalProcess.ToString();
                    processInfo.UseShellExecute = true;
                    // // processInfo.UserName = "azueruser";
                    // processInfo.WorkingDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);

                    // // processInfo.Password = ToSecureString("testtest123?");
                    // Process pro = new Process();
                    // pro.EnableRaisingEvents = true;
                    // pro.Exited += new EventHandler(process_Exited);
                    // pro.StartInfo = processInfo;
                    
                    // Console.WriteLine("Start env " + i.ToString());
                    
                    // pro.Start();
                    Process.Start(processInfo);
                }catch(Exception e){
                    Console.WriteLine(e.ToString());
                }
                //pro.WaitForExit(18000);
                curRunningProcess++;
                //如果大于最大并行数，就等待进程退出，是并行数不超过最大并行数
                // while (curRunningProcess >= maxParallelProcess)
                // {
                //     if (i >= totalProcess - 1)
                //     { return; }
                // }
            }
        }

        /// <summary>
        /// 进程结束
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
       private void process_Exited(object sender, EventArgs e)
        {
            curRunningProcess--;
        }
    }
}