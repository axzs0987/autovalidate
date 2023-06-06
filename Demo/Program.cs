using ExcelReader;
using System;
using System.IO;
using System.IO.Compression;
using System.Threading;
using ClosedXML.Excel;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json;
using XLParser;

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
        static void saveAsJson(object need_save_content, string file_name){
            string jsonData = JsonConvert.SerializeObject(need_save_content);
            File.WriteAllText(file_name, jsonData);
        }
        static void analyze_formula(string json_path, string data_set_path){
            string jsonstring1 = File.ReadAllText(json_path);
            List<string> sampled_file = JsonConvert.DeserializeObject<List<string>>(jsonstring1);
            Dictionary<string, List<SimpleCellFormula>> result = new Dictionary<string, List<SimpleCellFormula>>();
            int count = 0;
            int batch_id = 58;
            foreach(var fname in sampled_file){
                count += 1;
                string source_filename = data_set_path + "source_xlsx/" + fname;
                var workbook = new XLWorkbook(source_filename);
                var worksheets = workbook.Worksheets.ToArray();
                
                foreach(var sheet in worksheets){
                    string filesheet = fname + "---" + sheet;    
                    string sheetname = sheet.Name;
                    try{
                        int firstcolumn = sheet.FirstColumnUsed().ColumnNumber();
                        int lastcolumn = sheet.LastColumnUsed().ColumnNumber();
                        int firstrow = sheet.FirstRowUsed().RowNumber();
                        int lastrow = sheet.LastRowUsed().RowNumber();
                        List<SimpleCellFormula> forlist = new List<SimpleCellFormula>();
                        result.Add(filesheet, forlist);
                        for(int row =firstrow; row <= lastrow; row++){
                            for(int col = firstcolumn; col <= lastcolumn; col++){
                    
                                var cell = sheet.Cell(row, col);
                                if(cell.FormulaA1.Length==0){
                                    continue;
                                }
                            

                                SimpleCellFormula formula = new SimpleCellFormula();
                                formula.row = row;
                                formula.column = col;
                                formula.formulaR1C1 = cell.FormulaR1C1;
                                
                            
                                result[filesheet].Add(formula);
                            }
                        }
                    }
                    catch{
                        Console.WriteLine("error:no column or row.");
                        continue;
                    }
                }
                if(result.Count() >= 500){
                        saveAsJson(result, data_set_path + "save_formulas_jsons/origin_data_formulas.json");
                        // saveAsJson(result, "Formulas_20000sheets_recheck_"+batch_id.ToString()+".json");
                        batch_id += 1;
                        result = new Dictionary<string, List<SimpleCellFormula>>();
                    }
            }
            saveAsJson(result, data_set_path + "save_formulas_jsons/origin_data_formulas.json");
            // saveAsJson(result, "Formulas_20000sheets_recheck_"+batch_id.ToString()+".json");
        }
        static void Main(string[] args)
        {
//             extract features
             bool extract_all = true;
             string data_set_name = "ibm_data_set";
             string data_set_path = "../data_set/" + data_set_name + "/";
             if(extract_all){
                 // extract all xlsx features in one path
                 FeatureExtraction fe = new FeatureExtraction();
                 fe.extract_all_workbook(data_set_path + "source_xlsx/", data_set_path + "source_json/", data_set_path + "zip_files/xlsx_data_set.zip", "xlsx/");
             }
             else{
                 // extract one xlsx feature
                 string filename = "Excel - DataSet Test Examples - 0110.xlsx";
                 string entry = "test_data/";
                 string zip_file = "../test_data_0110.zip";
                 string save_path = "../tmp_data/workbook_features/";
                 extractExcel(filename, entry, zip_file, save_path);
             }
            // 调用 analyze_formula 方法
            string json_path = "../analyze-dv-1/" + data_set_name + "/"  + "file_names" + ".json";
            analyze_formula(json_path, data_set_path);
            // 可选：等待控制台按键后退出程序
            Console.WriteLine("end of feature ");
            Console.ReadKey();
        }
    }
}