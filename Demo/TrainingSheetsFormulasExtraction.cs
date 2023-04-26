using ExcelReader;
using System;
using System.IO;
using System.IO.Compression;

using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json;

namespace Demo
{
    class FeatureExtraction{
        public void ExtractOneSheet(string workbook_name, string zip_file_path, string entry_root_path, string save_path){
            string[] split_list = workbook_name.Split('/');
            string filename = workbook_name;
            var zipfile = zip_file_path;
 
            var entry = entry_root_path+ filename;
            using (var fs = File.Open(zipfile, FileMode.Open, FileAccess.Read))
            {
                var archive = new ZipArchive(fs, ZipArchiveMode.Read);
                try{
                    bool found = false;
                    foreach(var ent in archive.Entries){
                        if(ent.ToString() == entry){
                            found = true;
                        }
                    }
                    // Console.WriteLine(found);
                    using (var zs = archive.GetEntry(entry).Open())
                    {
                        var workbook = Worker.ExtractWorkbook(zs, entry);
                        File.WriteAllText(save_path + filename+".json", Worker.Jsonify(workbook));
                        workbook = null;
                        GC.Collect();
                        Console.WriteLine("success.");
                    }
                }catch{
                    return;
                }
                
            }
           
        }

        public void extract_all_workbook(string wokrbook_root_path, string save_path, string zip_file_path, string entry_root_path){
            DirectoryInfo root = new DirectoryInfo(wokrbook_root_path);
            FileInfo[] files=root.GetFiles();
            double all_time = 0;
            int wb_count = 0;

            foreach(var file in files){
                string wb_name = file.Name;
                wb_count += 1;
                if(File.Exists(save_path + wb_name + ".json")){
                    continue;
                }
              
                Console.WriteLine(wb_count.ToString()+'/' + files.Count().ToString());
                DateTime start_time = DateTime.Now;	//获取当前时间
                ExtractOneSheet(wb_name, zip_file_path, entry_root_path, save_path);
                DateTime end_time = DateTime.Now;	//获取当前时间
                TimeSpan ts = end_time - start_time;	//计算时间差
                double time = ts.TotalSeconds;	//将时间差转换为秒
                all_time += time;
            }
            
        }
    }
    
}