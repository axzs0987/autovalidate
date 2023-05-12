using System;
using ClosedXML.Excel;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Newtonsoft.Json;
using XLParser;

namespace AnalyzeDV
{
    class RefShift{
        public void saveAsJson(object need_save_content, string file_name){
            string jsonData = JsonConvert.SerializeObject(need_save_content);
            File.WriteAllText(file_name, jsonData);
        }

        public bool generate_tile_point(string rfilename, string filename, string sheetname){
            var worksheet = new XLWorkbook(rfilename).Worksheet(sheetname);
            var temp = worksheet.Protection;
            Console.WriteLine(temp.IsProtected);
            return temp.IsProtected;
            // int first_column = worksheet.FirstColumn().ColumnNumber();
            // int first_row = worksheet.FirstRow().RowNumber();
            // int last_column = worksheet.LastColumnUsed().ColumnNumber();
            // int last_row = worksheet.LastRowUsed().RowNumber();

            // Console.WriteLine("first_column:"+first_column.ToString());
            // Console.WriteLine("first_row:"+first_row.ToString());
            // Console.WriteLine("last_column:"+last_column.ToString());
            // Console.WriteLine("last_row:"+last_row.ToString());
            // List<List<int>> tile_range = new List<List<int>>();
            // List<int> row_list = new List<int>();
            // List<int> col_list = new List<int>();

            // row_list.Add(first_row);
            // row_list.Add(last_row);
            // col_list.Add(first_column);
            // col_list.Add(last_column);
            // tile_range.Add(row_list);
            // tile_range.Add(col_list);

            // List<int> row_nums = new List<int>();
            // var row_num = first_row;
            // while(row_num <= last_row){
            //     row_nums.Add(row_num);
            //     row_num += 100;
            // }

            // List<int> col_nums = new List<int>();
            // var col_num = first_column;
            // while(col_num <= last_column){
            //     col_nums.Add(col_num);
            //     col_num += 10;
            // }

            // // saveAsJson(row_nums, "/datadrive-2/data/fortune500_test/tile_rows/"+filename + "---" + sheetname + ".json");
            // // saveAsJson(col_nums, "/datadrive-2/data/fortune500_test/tile_cols/"+filename + "---" + sheetname + ".json");
            // // saveAsJson(tile_range, "/datadrive-2/data/fortune500_test/tile_range/"+filename + "---" + sheetname + ".json");
            // saveAsJson(row_nums, "/datadrive-2/data/top10domain_test/tile_rows/"+filename + "---" + sheetname + ".json");
            // saveAsJson(col_nums, "/datadrive-2/data/top10domain_test/tile_cols/"+filename + "---" + sheetname + ".json");
            // saveAsJson(tile_range, "/datadrive-2/data/top10domain_test/tile_range/"+filename + "---" + sheetname + ".json");
        }

        public void batch_get_tile_point()
        {
            // string path = "/datadrive-2/data/fortune500_test/afterfeature";
            // DirectoryInfo root = new DirectoryInfo(path);
            // FileInfo[] allFiles=root.GetFiles();
            string jsonstring = File.ReadAllText("../analyze-dv-1/training_ref_formulatokens.json");
            List<string> formula_tokens = JsonConvert.DeserializeObject<List<string>>(jsonstring);
            int count = 0;
            int prot = 0;
            // foreach (var file in allFiles)
            foreach (var f_token in formula_tokens)
            {
                
                // string formula_token = file.Name;
                // if(formula_token != "169915192308491905353010454443539152999-acceptedtable.xlsx---UPSInternational_Jan2021---412---18.npy"){
                    // continue;
                // }
                string formula_token = f_token;
                formula_token = formula_token.Replace("../../data/", "");
                string[] splited_token = formula_token.Split("---");
                string filename1 = splited_token[0];
                string filename = filename1.Split("/")[filename1.Split("/").Length-1];
                string sheetname = splited_token[1];

                Console.WriteLine(count.ToString() + "/" + formula_tokens.Count());
               
                // string rfilename = "/datadrive/data/crawled_xlsx_fortune500/" + filename;
                string rfilename = "/datadrive/data/" + filename1;
                // string filename = fname;
                Console.WriteLine("filename" + rfilename);
                try
                {
                //     // if (File.Exists("/datadrive-2/data/fortune500_test/tile_rows/"+filename + "---" + sheetname + ".json"))
                //     if (File.Exists("/datadrive-2/data/top10domain_test/tile_rows/"+filename + "---" + sheetname + ".json"))
                //     {
                //         continue;
                //     }
                    Console.WriteLine(filename);
                    Console.WriteLine(sheetname);
                    bool is_prote = generate_tile_point(rfilename, filename, sheetname);
                    if(is_prote){
                        prot += 1;
                    }
                    count += 1;
                    if(count == 200){
                        break;

                    }
                }
                catch
                {
                    Console.WriteLine("error");
                    continue;
                }
                // break;

            }
            Console.WriteLine("prot:" + prot.ToString());
            Console.WriteLine("all:" + count.ToString());
        }

    }
}