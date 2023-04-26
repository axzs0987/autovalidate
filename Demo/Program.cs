using ExcelReader;
using System;
using System.IO;
using System.IO.Compression;
using System.Threading;
namespace Demo
{
    class Program
    {
        static void extractExcel(string filename, string entry_path, string zipfile, string save_path){
            using (var fs = File.Open(zipfile, FileMode.Open, FileAccess.Read))
            {
                var archive = new ZipArchive(fs, ZipArchiveMode.Read);
                using (var zs = archive.GetEntry(entry_path + filename).Open())
                {
                    var workbook = Worker.ExtractWorkbook(zs, entry_path + filename);
                    File.WriteAllText(save_path + filename + ".json", Worker.Jsonify(workbook));
                }
            }
            Console.WriteLine("Extract Sccess!");
        }
    
        static void Main(string[] args)
        {
            bool extract_all = false;
            if(extract_all){
                // extract all xlsx features in one path
                FeatureExtraction fe = new FeatureExtraction();
                fe.extract_all_workbook("/datadrive/data_fuste/", "fuste_workbook_json/", "/datadrive/xls.zip", "xls/");
            }
            else{
                // extract one xlsx feature
                string filename = "Excel - Data Validation Examples - Reduced.xlsx";
                string entry = "test_data/";
                string zip_file = "../test_data.zip";
                string save_path = "../tmp_data/workbook_features/";
                extractExcel(filename, entry, zip_file, save_path);
            }
        }
    }
}