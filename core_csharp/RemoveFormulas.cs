using System;
using ClosedXML.Excel;
using System.Collections.Generic;
using System.Collections;
using System.Linq;
using System.IO;
using Newtonsoft.Json;
using XLParser;

namespace AnalyzeDV
{
    class RemoveFormulas{
        public void print_formulas(){
            string jsonstring = File.ReadAllText("sampled_file.json");
            List<string> sampled_file = JsonConvert.DeserializeObject<List<string>>(jsonstring);
            string jsonstring1 = File.ReadAllText("../analyze-dv-1/Formulas_77772sheets_mergerange_custom.json");
            Dictionary<string, Dictionary<string, Dictionary<string, SimpleFormula>>> formulas = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, Dictionary<string, SimpleFormula>>>>(jsonstring1);

            Dictionary<string, List<string>> workbook_worksheets = new Dictionary<string, List<string>>();
            foreach(string filesheet in sampled_file){
                string filename = filesheet.Split("---")[0];
                string sheetname = filesheet.Split("---")[1];
                if(!workbook_worksheets.ContainsKey(filename)){
                    List<string> sheetslist = new List<string>();
                    workbook_worksheets.Add(filename, sheetslist);
                }
                workbook_worksheets[filename].Add(sheetname);
            }

            int count = 0;
            foreach(string unzip_filepath in workbook_worksheets.Keys){
                try{
                    string filepath = unzip_filepath.Replace("UnzipData", "");
                    var workbook = new XLWorkbook(filepath);
                    string[] splitres = filepath.Split("/");
                    string filename = splitres[splitres.Count()-1];
                    if(! File.Exists("remove_formulas_200/"+filename)){
                        continue;
                    }
                    count += 1;
                    Console.WriteLine(count.ToString()+"/" + workbook_worksheets.Keys.Count().ToString());
                    foreach(string sheetname in workbook_worksheets[unzip_filepath]){
                        string filesheet = unzip_filepath + "---" + sheetname;
                        var worksheet = workbook.Worksheet(sheetname);
                        if(! formulas.ContainsKey(filesheet)){
                            continue;
                        }
                        foreach(string r1c1 in formulas[filesheet].Keys){
                            foreach(string id_ in formulas[filesheet][r1c1].Keys){
                                Console.WriteLine("filesheet:"+filesheet+", formula:"+formulas[filesheet][r1c1][id_].fr.ToString() +","+formulas[filesheet][r1c1][id_].fc.ToString() + ",r1c1:"+r1c1);
                                break;
                            }
                            break;
                        }
                        break;
                    }
                }catch{
                    continue;
                }
            }
        }
        public void remove_formulas(){
            string jsonstring = File.ReadAllText("sampled_file.json");
            List<string> sampled_file = JsonConvert.DeserializeObject<List<string>>(jsonstring);
            string jsonstring1 = File.ReadAllText("../analyze-dv-1/Formulas_77772sheets_mergerange_custom.json");
            Dictionary<string, Dictionary<string, Dictionary<string, SimpleFormula>>> formulas = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, Dictionary<string, SimpleFormula>>>>(jsonstring1);

            Dictionary<string, List<string>> workbook_worksheets = new Dictionary<string, List<string>>();
            foreach(string filesheet in sampled_file){
                string filename = filesheet.Split("---")[0];
                string sheetname = filesheet.Split("---")[1];
                if(!workbook_worksheets.ContainsKey(filename)){
                    List<string> sheetslist = new List<string>();
                    workbook_worksheets.Add(filename, sheetslist);
                }
                workbook_worksheets[filename].Add(sheetname);
            }

            int count = 0;
            foreach(string unzip_filepath in workbook_worksheets.Keys){
                try{
                    string filepath = unzip_filepath.Replace("UnzipData", "");
                    var workbook = new XLWorkbook(filepath);
                    string[] splitres = filepath.Split("/");
                    string filename = splitres[splitres.Count()-1];
                    if(File.Exists("remove_formulas/"+filename)){
                        continue;
                    }
                    count += 1;
                    Console.WriteLine(count.ToString()+"/" + workbook_worksheets.Keys.Count().ToString());
                    foreach(string sheetname in workbook_worksheets[unzip_filepath]){
                        string filesheet = unzip_filepath + "---" + sheetname;
                        var worksheet = workbook.Worksheet(sheetname);
                        if(! formulas.ContainsKey(filesheet)){
                            continue;
                        }
                        foreach(string r1c1 in formulas[filesheet].Keys){
                            foreach(string id_ in formulas[filesheet][r1c1].Keys){
                                SimpleFormula one_formula = formulas[filesheet][r1c1][id_];
                                // Console.WriteLine("fr:"+one_formula.fr.ToString()+", fc:"+one_formula.fc.ToString()+", lr:"+one_formula.lr.ToString()+", lc:"+one_formula.lc.ToString());
                                for(int row = one_formula.fr; row <= one_formula.lr; row ++){
                                    for(int col = one_formula.fc; col <= one_formula.lc; col++){
                                        // Console.WriteLine("row:"+row.ToString()+", col:"+col.ToString());
                                        worksheet.Cell(row, col).SetValue("");
                                        worksheet.Cell(row, col).SetFormulaA1("");
                                        worksheet.Cell(row, col).SetFormulaR1C1("");
                                    }
                                }
                            }
                        }
                    }
                    workbook.SaveAs("remove_formulas/"+filename);

                }catch{
                    continue;
                }
                
            }
        }
    }
}