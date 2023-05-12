using System;
using ClosedXML.Excel;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Newtonsoft.Json;
using XLParser;
using System.Drawing;
using System.Diagnostics;
namespace AnalyzeDV
{

    class FormulaSheetFeatures
    {
        public void saveAsJson(object need_save_content, string file_name)
        {
            string jsonData = JsonConvert.SerializeObject(need_save_content);
            File.WriteAllText(file_name, jsonData);
        }
        public int column_id(string column_cha)
        {
            int index = 1;
            int result = 0;
            while (index <= column_cha.Count())
            {
                int cha_num = (int)column_cha[column_cha.Count() - index] - 64;
                int sub_index = 0;
                int di = 1;
                while (sub_index < index - 1)
                {
                    di *= 26;
                    sub_index += 1;
                }
                result += di * cha_num;
                index += 1;
            }
            return result;
        }

        public List<int> RangeAdress2num(string dv_range)
        {
            List<int> result = new List<int>();
            string[] split_dv_range = dv_range.Split(':');
            string range_start_cell = split_dv_range[0];
            string range_end_cell = split_dv_range[1];
            int range_start_number_index = 0;
            foreach (var cha in range_start_cell)
            {
                if (!Char.IsNumber(cha))
                {
                    range_start_number_index += 1;
                    continue;
                }
                break;
            }
            int range_end_number_index = 0;
            foreach (var cha in range_end_cell)
            {
                if (!Char.IsNumber(cha))
                {
                    range_end_number_index += 1;
                    continue;
                }
                break;
            }

            string lty_c = range_start_cell.Substring(0, range_start_number_index);
            string ltx = range_start_cell.Substring(range_start_number_index);
            string rby_c = range_end_cell.Substring(0, range_end_number_index);
            string rbx = range_end_cell.Substring(range_end_number_index);

            // ltx, lty, rbx, rby
            result.Add(int.Parse(ltx));
            result.Add(column_id(lty_c));
            result.Add(int.Parse(rbx));
            result.Add(column_id(rby_c));
            return result;
        }
        public static string get_cell_content_template(IXLCell cell)
        {
            string result = "";
            if (cell.DataType == XLDataType.Text)
            {
                foreach (var cha in cell.Value.ToString())
                {
                    result += 'S';
                }
            }
            else if (cell.DataType == XLDataType.Number)
            {
                foreach (var cha in cell.Value.ToString())
                {
                    result += 'N';
                }
            }
            else if (cell.DataType == XLDataType.Boolean)
            {
                foreach (var cha in cell.Value.ToString())
                {
                    result += 'B';
                }
            }
            else if (cell.DataType == XLDataType.DateTime)
            {
                foreach (var cha in cell.Value.ToString())
                {
                    result += 'D';
                }
            }
            else if (cell.DataType == XLDataType.TimeSpan)
            {
                foreach (var cha in cell.Value.ToString())
                {
                    result += 'T';
                }
            }
            return result;
        }
        public static Color get_color_from_one_xlcolor(XLWorkbook workbook, XLColor xlcolor)
        {
            Color color = new Color();
            if (xlcolor.ColorType == XLColorType.Color)
            {
                return xlcolor.Color;
            }
            if (xlcolor.ColorType == XLColorType.Theme)
            {
                if (xlcolor.ThemeColor == XLThemeColor.Background1)
                {
                    return workbook.Theme.Background1.Color;
                }
                if (xlcolor.ThemeColor == XLThemeColor.Text1)
                {
                    return workbook.Theme.Text1.Color;
                }
                if (xlcolor.ThemeColor == XLThemeColor.Background2)
                {
                    return workbook.Theme.Background2.Color;
                }
                if (xlcolor.ThemeColor == XLThemeColor.Text2)
                {
                    return workbook.Theme.Text2.Color;
                }
                if (xlcolor.ThemeColor == XLThemeColor.Accent1)
                {
                    return workbook.Theme.Accent1.Color;
                }
                if (xlcolor.ThemeColor == XLThemeColor.Accent2)
                {
                    return workbook.Theme.Accent2.Color;
                }
                if (xlcolor.ThemeColor == XLThemeColor.Accent3)
                {
                    return workbook.Theme.Accent3.Color;
                }
                if (xlcolor.ThemeColor == XLThemeColor.Accent4)
                {
                    return workbook.Theme.Accent4.Color;
                }
                if (xlcolor.ThemeColor == XLThemeColor.Accent5)
                {
                    return workbook.Theme.Accent5.Color;
                }
                if (xlcolor.ThemeColor == XLThemeColor.Accent6)
                {
                    return workbook.Theme.Accent6.Color;
                }
                if (xlcolor.ThemeColor == XLThemeColor.Hyperlink)
                {
                    return workbook.Theme.Hyperlink.Color;
                }
                if (xlcolor.ThemeColor == XLThemeColor.FollowedHyperlink)
                {
                    return workbook.Theme.FollowedHyperlink.Color;
                }
                // Console.WriteLine("xlcolor.ThemeColor:" +xlcolor.ThemeColor.ToString());
                return xlcolor.Color;
            }
            else
            {
                // Console.WriteLine("Index color");
                // Console.WriteLine("no color");
                return xlcolor.Color;
            }
        }
        public static CellFeature get_cell_features(XLWorkbook workbook, IXLCell cell)
        {
            CellFeature cell_feature = new CellFeature();
            Color background_color = get_color_from_one_xlcolor(workbook, cell.Style.Fill.BackgroundColor);
            Color font_color = get_color_from_one_xlcolor(workbook, cell.Style.Font.FontColor);
            cell_feature.background_color_r = background_color.R;
            // Console.WriteLine(cell_feature.background_color_r);
            cell_feature.background_color_g = background_color.G;
            cell_feature.background_color_b = background_color.B;
            cell_feature.font_color_r = font_color.R;
            cell_feature.font_color_g = font_color.G;
            cell_feature.font_color_b = font_color.B;
            cell_feature.font_size = cell.Style.Font.FontSize;
            cell_feature.font_ita = cell.Style.Font.Italic;
            cell_feature.font_bold = cell.Style.Font.Bold;
            cell_feature.font_strikethrough = cell.Style.Font.Strikethrough;
            cell_feature.font_shadow = cell.Style.Font.Shadow;
            cell_feature.height = cell.WorksheetRow().Height;
            cell_feature.width = cell.WorksheetColumn().Width;
            // cell_feature.content = cell.Value.ToString();
            // cell_feature.content_template = get_cell_content_template(cell);
            return cell_feature;
        }

        public static CellFeature get_other_cell_features(XLWorkbook workbook, IXLCell cell)
        {
            CellFeature cell_feature = new CellFeature();
            Color background_color = get_color_from_one_xlcolor(workbook, cell.Style.Fill.BackgroundColor);
            Color font_color = get_color_from_one_xlcolor(workbook, cell.Style.Font.FontColor);
            cell_feature.background_color_r = background_color.R;
            // Console.WriteLine(cell_feature.background_color_r);
            cell_feature.background_color_g = background_color.G;
            cell_feature.background_color_b = background_color.B;
            cell_feature.font_color_r = font_color.R;
            cell_feature.font_color_g = font_color.G;
            cell_feature.font_color_b = font_color.B;
            cell_feature.font_size = cell.Style.Font.FontSize;
            cell_feature.font_ita = cell.Style.Font.Italic;
            cell_feature.font_bold = cell.Style.Font.Bold;
            cell_feature.font_strikethrough = cell.Style.Font.Strikethrough;
            cell_feature.font_shadow = cell.Style.Font.Shadow;
            cell_feature.height = cell.WorksheetRow().Height;
            cell_feature.width = cell.WorksheetColumn().Width;
            // cell_feature.content = cell.Value.ToString();
            // cell_feature.content_template = get_cell_content_template(cell);
            return cell_feature;
        }


        public void batch_get_dv_sheet_features()
        {
            // string jsonstring1 = File.ReadAllText("../analyze-dv-1/Formulas_20000sheets_mergerange_custom.json");
            // string jsonstring1 = File.ReadAllText("../analyze-dv-1/Formulas_77772sheets_mergerange_custom.json");
            // string jsonstring1 = File.ReadAllText("../analyze-dv-1/origin_top10domain_mergerange_new_res_1.json");
            // string jsonstring1 = File.ReadAllText("../analyze-dv-1/origin_middle10domain_mergerange_new_res_1.json");
            string jsonstring1 = File.ReadAllText("../analyze-dv-1/origin_fortune500_mergerange_new_res_1.json");
            // filename : r1c1: batchid: Formula
            var formulas = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, Dictionary<string, SimpleFormula>>>>(jsonstring1);

            List<string> result = new List<string>();
            // Stopwatch stop_watch = new Stopwatch();
            int count = 0;
            string save_path = "";
            foreach (string filesheet in formulas.Keys)
            {
                count += 1;
                // if(count < (formulas.Keys.Count()/6)*threadid || count >= (formulas.Keys.Count()/6)*(threadid+1)){
                //     continue;
                // }

                // stop_watch.Start();
                string[] filesheetsplit = filesheet.Split("---");
                string filename = filesheetsplit[0];
                string sheetname = filesheetsplit[1];
                string[] filenames = filename.Split('/');
                string filename1 = filenames[filenames.Count() - 1];
                // if(filename1 + "---" + sheetname == "f941cd3a-7a93-4932-8846-ce7dcba7019b_dC1zb3VyY2UvcHJvZmVzc2lvbmFsLWRldmVsb3BtZW50LzIwMTktMjAyMC1tZW50b3ItdGVhY2hlci1hbmQtY29udGVudC1sZWFkZXItbm9taW5hdGlvbi10ZW1wbGF0ZS54bHN4P3NmdnJzbj04MzcwOWUxZl84.xlsx"){
                // continue;
                // }
                // Console.WriteLine(filesheet);
                // Stopwatch stop_watch1 = new Stopwatch();
                // stop_watch1.Start();
                filename = filename.Replace("/UnzipData", "");
                try
                {
                    var workbook = new XLWorkbook("/datadrive/data_fortune500/crawled_xlsx_fortune500/" + filename);
                    var worksheet = workbook.Worksheet(sheetname);
                    int ws_first_column = worksheet.FirstColumn().ColumnNumber();
                    int ws_last_column = worksheet.LastColumn().ColumnNumber();
                    int ws_first_row = worksheet.FirstRow().RowNumber();
                    int ws_last_row = worksheet.LastRow().RowNumber();
                    // stop_watch1.Stop();
                    foreach (string r1c1 in formulas[filesheet].Keys)
                    {
                        foreach (string batch_id in formulas[filesheet][r1c1].Keys)
                        {
                            SimpleFormula formula = formulas[filesheet][r1c1][batch_id];
                            int first_row = formula.fr;
                            int first_column = formula.fc;
                            // Console.WriteLine(filename);
                            // if(File.Exists("/datadrive-2/data/origin_top10domain_formula_features/" + filename.Split("/")[filename.Split("/").Count()-1] + "---" + sheetname + "---" + first_row.ToString() + "---" + first_column.ToString() + ".json") ){
                            // if(File.Exists("/datadrive-2/data/middle10domain_test/origin_middle10domain_formula_features/" + filename.Split("/")[filename.Split("/").Count()-1] + "---" + sheetname + "---" + first_row.ToString() + "---" + first_column.ToString() + ".json") ){
                            if (File.Exists("/datadrive-2/data/fortune500_test/origin_fortune500_formula_features/" + filename.Split("/")[filename.Split("/").Count() - 1] + "---" + sheetname + "---" + first_row.ToString() + "---" + first_column.ToString() + ".json"))
                            {
                                // Console.WriteLine("Exists.");
                                continue;
                            }
                            // Console.WriteLine("read sheet time:"+stop_watch1.ElapsedMilliseconds.ToString() + " ms");
                            get_sheet_feature(filename, sheetname, workbook, worksheet, first_row, first_column, ws_first_column, ws_last_column, ws_first_row, ws_last_row, save_path);

                        }
                    }
                }
                catch
                {
                    continue;
                }

                // stop_watch.Stop();
                // StreamWriter streamWriter = new StreamWriter("ExtractAllFormulaTime/" + filename.Split("/")[4] + "---" + sheetname +".txt", true);
                // streamWriter.Write(stop_watch.ElapsedMilliseconds.ToString() + " ms");
                // streamWriter.Close();
            }

        }
        public void batch_get_another_sheet_features()
        {
            // string jsonstring1 = File.ReadAllText("../analyze-dv-1/Formulas_20000sheets_mergerange_custom.json");
            string jsonstring1 = File.ReadAllText("../analyze-dv-1/TrainingFormulas_mergerange_custom_new_res_1.json");
            // filename : r1c1: batchid: Formula
            var formulas = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, Dictionary<string, SimpleFormula>>>>(jsonstring1);

            List<string> result = new List<string>();
            // Stopwatch stop_watch = new Stopwatch();
            int count = 0;
            foreach (string filesheet in formulas.Keys)
            {
                count += 1;
                if (count <= 24622)
                {
                    continue;
                }
                string[] filesheetsplit = filesheet.Split("---");
                string filename = filesheetsplit[0];
                string sheetname = filesheetsplit[1];
                string[] filenames = filename.Split('/');
                string filename1 = filenames[filenames.Count() - 1];
                // if(filename1 + "---" + sheetname == "f941cd3a-7a93-4932-8846-ce7dcba7019b_dC1zb3VyY2UvcHJvZmVzc2lvbmFsLWRldmVsb3BtZW50LzIwMTktMjAyMC1tZW50b3ItdGVhY2hlci1hbmQtY29udGVudC1sZWFkZXItbm9taW5hdGlvbi10ZW1wbGF0ZS54bHN4P3NmdnJzbj04MzcwOWUxZl84.xlsx"){
                // continue;
                // }
                Console.WriteLine(count.ToString() + '/' + formulas.Keys.Count().ToString());
                // Stopwatch stop_watch1 = new Stopwatch();
                // stop_watch1.Start();
                if (sheetname == "")
                {
                    continue;
                }
                filename = filename.Replace("/UnzipData", "");
                var workbook = new XLWorkbook(filename);
                var worksheet = workbook.Worksheet(sheetname);
                int ws_first_column = worksheet.FirstColumn().ColumnNumber();
                int ws_last_column = worksheet.LastColumn().ColumnNumber();
                int ws_first_row = worksheet.FirstRow().RowNumber();
                int ws_last_row = worksheet.LastRow().RowNumber();
                // stop_watch1.Stop();
                foreach (string r1c1 in formulas[filesheet].Keys)
                {
                    foreach (string batch_id in formulas[filesheet][r1c1].Keys)
                    {
                        SimpleFormula formula = formulas[filesheet][r1c1][batch_id];
                        int first_row = formula.fr;
                        int first_column = formula.fc;
                        // Console.WriteLine(filename);
                        if (File.Exists("../analyze-dv-1/other_training_formulas/" + filename.Split("/")[filename.Split("/").Count() - 1] + "---" + sheetname + "---" + first_row.ToString() + "---" + first_column.ToString() + ".json") || File.Exists("../analyze-dv-1/formulas196/" + filename.Split("/")[filename.Split("/").Count() - 1] + "---" + sheetname + "---" + first_row.ToString() + "---" + first_column.ToString() + ".json"))
                        {
                            // Console.WriteLine("Exists.");
                            continue;
                        }
                        // Console.WriteLine("read sheet time:"+stop_watch1.ElapsedMilliseconds.ToString() + " ms");
                        get_other_sheet_feature(filename, sheetname, workbook, worksheet, first_row, first_column, ws_first_column, ws_last_column, ws_first_row, ws_last_row);

                    }
                }
                // stop_watch.Stop();
                // StreamWriter streamWriter = new StreamWriter("ExtractAllFormulaTime/" + filename.Split("/")[4] + "---" + sheetname +".txt", true);
                // streamWriter.Write(stop_watch.ElapsedMilliseconds.ToString() + " ms");
                // streamWriter.Close();
            }

        }

        public void batch_get_neighbors_features()
        {
            // string jsonstring1 = File.ReadAllText("../analyze-dv-1/Formulas_20000sheets_mergerange_custom.json");
            string jsonstring1 = File.ReadAllText("../analyze-dv-1/negative_neighbors_dict.json");
            // filename : r1c1: batchid: Formula
            Dictionary<string, List<string>> neighbors = JsonConvert.DeserializeObject<Dictionary<string, List<string>>>(jsonstring1);

            List<string> result = new List<string>();
            // Stopwatch stop_watch = new Stopwatch();
            int count = 0;
            foreach (string filesheet in neighbors.Keys)
            {
                count += 1;
                string[] filesheetsplit = filesheet.Split("---");
                string filename = filesheetsplit[0];
                string sheetname = filesheetsplit[1];
                string[] filenames = filename.Split('/');
                string filename1 = filenames[filenames.Count() - 1];
                Console.WriteLine(count.ToString() + '/' + neighbors.Count().ToString());

                Stopwatch stop_watch1 = new Stopwatch();
                stop_watch1.Start();

                if (sheetname == "")
                {
                    continue;
                }

                filename = filename.Replace("/UnzipData", "");
                var workbook = new XLWorkbook(filename);
                var worksheet = workbook.Worksheet(sheetname);
                stop_watch1.Stop();
                Console.WriteLine("read sheet time:" + stop_watch1.ElapsedMilliseconds.ToString() + " ms");
                foreach (string token in neighbors[filesheet])
                {
                    string[] tokenlist = token.Split("---");
                    string fr_str = tokenlist[2];
                    string fc_str = tokenlist[3];


                    Stopwatch stop_watch = new Stopwatch();

                    int first_row = int.Parse(fr_str);
                    int first_column = int.Parse(fc_str);
                    if (File.Exists("/datadrive-2/data/neighbors/" + filename.Split("/")[filename.Split("/").Count() - 1] + "---" + sheetname + "---" + first_row.ToString() + "---" + first_column.ToString() + ".json"))
                    {
                        Console.WriteLine("Exists.");
                        continue;
                    }



                    int ws_first_column = worksheet.FirstColumn().ColumnNumber();
                    int ws_last_column = worksheet.LastColumn().ColumnNumber();
                    int ws_first_row = worksheet.FirstRow().RowNumber();
                    int ws_last_row = worksheet.LastRow().RowNumber();
                    // Console.WriteLine(filename);


                    stop_watch.Start();
                    get_neighbors(filename, sheetname, workbook, worksheet, first_row, first_column, ws_first_column, ws_last_column, ws_first_row, ws_last_row);
                    stop_watch.Stop();
                    Console.WriteLine("extract feature time:" + stop_watch.ElapsedMilliseconds.ToString() + " ms");

                }

            }

        }

        // public void batch_get_sheet_features(){

        //     string jsonstring1 = File.ReadAllText("../analyze-dv-1/Formulas_20000sheets_mergerange_custom.json");
        //     // filename : r1c1: batchid: Formula
        //     var formulas = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, Dictionary<string, Formula>>>>(jsonstring1);

        //     List<string> result = new List<string>();

        //     int count = 0;
        //     foreach(string filesheet in formulas.Keys){
        //         string[] filesheetsplit = filesheet.Split("---");
        //         string filename = filesheetsplit[0];
        //         string sheetname = filesheetsplit[1];
        //         string[] filenames = filename.Split('/');
        //         string filename1 = filenames[filenames.Count() - 1];
        //         if(filename1=="4051c92706de3dc131bdc892985347e6_d3d3Lmdvb2RseS5jby5pbgkxMDQuMTguMzUuNjI=.xlsx" || filename1=="5732acf19842ca9ef8b25a3b3f881381_YWdpbGVjb25zb3J0aXVtLnBid29ya3MuY29tCTIwOC45Ni4xOC4yMzg=.xlsx"){
        //             continue;
        //         }
        //         if(File.Exists("../analyze-dv-1/filename2bertfeature/" + filename.Split('/')[4] + "---" + sheetname + ".npy")){
        //             continue;
        //         }
        //         // foreach(string r1c1 in formulas[filesheet].Keys){
        //             // foreach(string batch_id in formulas[filesheet][r1c1].Keys){
        //                 // Formula formula = formulas[filesheet][r1c1][batch_id];
        //                 // int fr = formula.fr;
        //                 // int fc = formula.fc;
        //                 // Console.WriteLine("fr:"+fr.ToString());
        //                 // Console.WriteLine("fc:"+fc.ToString());
        //                 // Console.WriteLine("lr:"+formula.lr.ToString());
        //                 // Console.WriteLine("lc:"+formula.lc.ToString());
        //                 try{
        //                     Console.WriteLine("get_sheet_feature......");
        //                     get_sheet_feature_0100(filename, sheetname);
        //                 }catch{
        //                     continue;
        //                 }
        //             // }    
        //         // }
        //     }

        // }
        public void batch_get_sheet_features()
        {
            // string jsonstring1 = File.ReadAllText("../analyze-dv-1/Formulas_20000sheets_mergerange_custom.json");
            // string jsonstring1 = File.ReadAllText("../analyze-dv-1/origin_top10domain_filesheets.json");
            // string jsonstring1 = File.ReadAllText("../analyze-dv-1/origin_middle10domain_filesheets.json");
            string jsonstring1 = File.ReadAllText("../analyze-dv-1/origin_fortune500_filesheets.json");
            // filename : r1c1: batchid: Formula
            var filesheets = JsonConvert.DeserializeObject<Dictionary<string, List<string>>>(jsonstring1);
            Console.WriteLine("succeed loading data.....");
            List<string> result = new List<string>();
            // Stopwatch stop_watch = new Stopwatch();
            int count = 0;
            string save_path = "";
            foreach (string fname in filesheets.Keys)
            {
                count += 1;
                Console.WriteLine(count.ToString() + "/" + filesheets.Keys.Count());
                Stopwatch stop_watch1 = new Stopwatch();
                stop_watch1.Start();
                // string filename = "/datadrive/data/"+fname;
                string filename = "/datadrive/data_fortune500/crawled_xlsx_fortune500/" + fname;
                // string filename = fname;
                Console.WriteLine("filename" + filename);
                try
                {
                    var workbook = new XLWorkbook(filename);
                    stop_watch1.Stop();
                    Console.WriteLine("read sheet time:" + stop_watch1.ElapsedMilliseconds.ToString() + " ms");
                    foreach (string sheetname in filesheets[fname])
                    {
                        var worksheet = workbook.Worksheet(sheetname);
                        var ws_first_column = worksheet.FirstColumn().ColumnNumber();
                        var ws_last_column = worksheet.LastColumn().ColumnNumber();
                        var ws_first_row = worksheet.FirstRow().RowNumber();
                        var ws_last_row = worksheet.LastRow().RowNumber();
                        // if(File.Exists("/datadrive-2/data/middle10domain_test/middle10domain_origin_sheet_features/" + filename.Split("/")[filename.Split("/").Count()-1] + "---" + sheetname +  ".json") ){
                        if (File.Exists("/datadrive-2/data/fortune500_test/origin_sheet_features/" + filename.Split("/")[filename.Split("/").Count() - 1] + "---" + sheetname + ".json"))
                        {
                            // Console.WriteLine("Exists.");
                            continue;
                        }
                        get_sheet_feature(filename, sheetname, workbook, worksheet, 1, 1, ws_first_column, ws_last_column, ws_first_row, ws_last_row, save_path);
                    }
                }
                catch
                {
                    continue;
                }


            }

        }
        public void get_sheet_feature(string filename, string sheetname, XLWorkbook workbook, IXLWorksheet worksheet, int fr, int fc, int ws_fc, int ws_lc, int ws_fr, int ws_lr, string save_path)
        {

            string[] filenames = filename.Split('/');
            string filename1 = filenames[filenames.Count() - 1];

            // Console.WriteLine(filename1 + "---" + sheetname + ".json");
            SheetFeature sheet_feature = new SheetFeature();
            sheet_feature.filename = filename;
            sheet_feature.sheetname = sheetname;
            filename = filename.Replace("/UnzipData", "");

            List<CellFeature> one_sheet_feature = new List<CellFeature>();
            Stopwatch stop_watch1 = new Stopwatch();
            stop_watch1.Start();
            int ltx = fr;
            int lty = fc;
            int start_row = ltx - 50 > 1 ? ltx - 50 : 1;
            int start_column = lty - 5 > 1 ? lty - 5 : 1;
            List<int> start_row_column = new List<int>();
            start_row_column.Add(start_row);
            start_row_column.Add(start_column);

            for (int row = start_row; row <= start_row + 99; row += 1)
            {
                for (int column = start_column; column <= start_column + 9; column += 1)
                {
                    CellFeature cell_feature = new CellFeature();
                    if (column >= ws_fc && column <= ws_lc && row >= ws_fr && row <= ws_lr)
                    {
                        IXLCell cell1 = worksheet.Cell(row, column);
                        // Console.WriteLine(row.ToString() + "," + column.ToString());
                        // Console.WriteLine(row.ToString()+ ',' + column.ToString());
                        if (!cell1.IsEmpty())
                        {
                            // Console.WriteLine(cell1.Value.GetType());
                            // bool catched = cell1.TryGetValue<string>(out string value1);
                            // if(catched){
                            // Console.WriteLine("Catched");
                            cell_feature = get_cell_features(workbook, cell1);

                        }

                    }
                    one_sheet_feature.Add(cell_feature);
                }
            }
            stop_watch1.Stop();
            // Console.WriteLine("iterate time:" + stop_watch1.ElapsedMilliseconds.ToString() + " ms");
            sheet_feature.sheetfeature = one_sheet_feature;
            string jsonData = JsonConvert.SerializeObject(sheet_feature);
            // Console.WriteLine("Save formula features:" + filename.Split("/")[filename.Split("/").Count() - 1] + "---" + sheetname + ".json");
            // File.WriteAllText("/datadrive-2/data/origin_top10domain_formula_features/" + filename.Split("/")[1] + "---" + sheetname + "---" + fr.ToString() + "---" + fc.ToString() + ".json", jsonData);
            // File.WriteAllText("/datadrive-2/data/middle10domain_test/origin_middle10domain_formula_features/" + filename.Split("/")[1] + "---" + sheetname + "---" + fr.ToString() + "---" + fc.ToString() + ".json", jsonData);
            // File.WriteAllText("/datadrive-2/data/fortune500_test/origin_sheet_features/" + filename.Split("/")[filename.Split("/").Count() - 1] + "---" + sheetname + ".json", jsonData);
            File.WriteAllText(save_path + "/" + filename.Split("/")[filename.Split("/").Count() - 1] + "---" + sheetname + "---" + fr.ToString() + "---" + fc.ToString() + ".json", jsonData);
            // File.WriteAllText("/datadrive-2/data/middle10domain_test/middle10domain_origin_sheet_features/" + filename.Split("/")[4] + "---" + sheetname + "---" + fr.ToString() + "---" + fc.ToString() + ".json", jsonData);

        }

        public void get_other_sheet_feature(string filename, string sheetname, XLWorkbook workbook, IXLWorksheet worksheet, int fr, int fc, int ws_fc, int ws_lc, int ws_fr, int ws_lr)
        {

            string[] filenames = filename.Split('/');
            string filename1 = filenames[filenames.Count() - 1];

            // Console.WriteLine(filename1 + "---" + sheetname + ".json");
            SheetFeature sheet_feature = new SheetFeature();
            sheet_feature.filename = filename;
            sheet_feature.sheetname = sheetname;
            filename = filename.Replace("/UnzipData", "");

            List<CellFeature> one_sheet_feature = new List<CellFeature>();
            Stopwatch stop_watch1 = new Stopwatch();
            stop_watch1.Start();
            int ltx = fr;
            int lty = fc;
            int start_row = ltx - 50 > 1 ? ltx - 50 : 1;
            int start_column = lty - 5 > 1 ? lty - 5 : 1;
            List<int> start_row_column = new List<int>();
            start_row_column.Add(start_row);
            start_row_column.Add(start_column);

            for (int row = start_row; row <= start_row + 99; row += 1)
            {
                for (int column = start_column; column <= start_column + 9; column += 1)
                {
                    CellFeature cell_feature = new CellFeature();
                    if (column >= ws_fc && column <= ws_lc && row >= ws_fr && row <= ws_lr)
                    {
                        IXLCell cell1 = worksheet.Cell(row, column);
                        // Console.WriteLine(row.ToString() + "," + column.ToString());
                        // Console.WriteLine(row.ToString()+ ',' + column.ToString());
                        if (!cell1.IsEmpty())
                        {
                            cell_feature = get_other_cell_features(workbook, cell1);
                        }

                    }
                    one_sheet_feature.Add(cell_feature);
                }
            }
            stop_watch1.Stop();
            // Console.WriteLine("iterate time:"+stop_watch1.ElapsedMilliseconds.ToString() + " ms");
            sheet_feature.sheetfeature = one_sheet_feature;
            string jsonData = JsonConvert.SerializeObject(sheet_feature);
            // Console.WriteLine("Save formula features:"+ filename + "---" + sheetname + "---" + fr.ToString() + "---" + fc.ToString());
            File.WriteAllText("../analyze-dv-1/other_training_formulas/" + filename.Split("/")[4] + "---" + sheetname + "---" + fr.ToString() + "---" + fc.ToString() + ".json", jsonData);

        }
        public void get_neighbors(string filename, string sheetname, XLWorkbook workbook, IXLWorksheet worksheet, int fr, int fc, int ws_fc, int ws_lc, int ws_fr, int ws_lr)
        {

            string[] filenames = filename.Split('/');
            string filename1 = filenames[filenames.Count() - 1];

            // Console.WriteLine(filename1 + "---" + sheetname + ".json");
            SheetFeature sheet_feature = new SheetFeature();
            sheet_feature.filename = filename;
            sheet_feature.sheetname = sheetname;
            filename = filename.Replace("/UnzipData", "");

            List<CellFeature> one_sheet_feature = new List<CellFeature>();
            Stopwatch stop_watch1 = new Stopwatch();
            stop_watch1.Start();
            int ltx = fr;
            int lty = fc;
            int start_row = ltx - 50 > 1 ? ltx - 50 : 1;
            int start_column = lty - 5 > 1 ? lty - 5 : 1;
            List<int> start_row_column = new List<int>();
            start_row_column.Add(start_row);
            start_row_column.Add(start_column);

            for (int row = start_row; row <= start_row + 99; row += 1)
            {
                for (int column = start_column; column <= start_column + 9; column += 1)
                {
                    CellFeature cell_feature = new CellFeature();
                    if (column >= ws_fc && column <= ws_lc && row >= ws_fr && row <= ws_lr)
                    {
                        IXLCell cell1 = worksheet.Cell(row, column);
                        // Console.WriteLine(row.ToString() + "," + column.ToString());
                        // Console.WriteLine(row.ToString()+ ',' + column.ToString());
                        if (!cell1.IsEmpty())
                        {
                            cell_feature = get_other_cell_features(workbook, cell1);
                        }

                    }
                    one_sheet_feature.Add(cell_feature);
                }
            }
            stop_watch1.Stop();
            // Console.WriteLine("iterate time:"+stop_watch1.ElapsedMilliseconds.ToString() + " ms");
            sheet_feature.sheetfeature = one_sheet_feature;
            string jsonData = JsonConvert.SerializeObject(sheet_feature);
            // Console.WriteLine("Save formula features:"+ filename + "---" + sheetname + "---" + fr.ToString() + "---" + fc.ToString());
            File.WriteAllText("/datadrive-2/data/neighbors/" + filename.Split("/")[4] + "---" + sheetname + "---" + fr.ToString() + "---" + fc.ToString() + ".json", jsonData);

        }

        public void batch_generate_deep_tile_features(){
            DirectoryInfo root = new DirectoryInfo("/datadrive-2/data/fortune500_test/second_tile_position/");
            FileInfo[] files=root.GetFiles();
            List<string> result = new List<string>();
            string save_path = "/datadrive-2/data/fortune500_test/second_tile_features/";
            foreach(var fileinfo in files){
                Stopwatch stop_watch = new Stopwatch();
                stop_watch.Start();
                string origin_filename = fileinfo.Name;
                
                string[] splited_origin_filenames = origin_filename.Split(".json")[0].Split("---");
                string filename = splited_origin_filenames[0];
                string sheetname = splited_origin_filenames[1];

                string jsonstring1 = File.ReadAllText("/datadrive-2/data/fortune500_test/second_tile_position/"+origin_filename);
                var tile_cols_rows = JsonConvert.DeserializeObject<List<List<int>>>(jsonstring1);
                var tile_cols = tile_cols_rows[0];
                var tile_rows = tile_cols_rows[1];
                var is_all_exists = true;
                foreach(int col in tile_cols){
                        foreach(int row in tile_rows){
            
                                if (!File.Exists(save_path + "/" + filename.Split("/")[filename.Split("/").Count() - 1] + "---" + sheetname + "---" + row.ToString() + "---" + col.ToString() + ".json"))
                                {
                                    is_all_exists = false;
                                    break;
                                }
                        }
                }
                if(is_all_exists){
                    Console.WriteLine("all exists.");
                    continue;
                }
                Console.WriteLine("origin_filename:"+origin_filename);
                try{
                    var workbook = new XLWorkbook("/datadrive/data_fortune500/crawled_xlsx_fortune500/" + filename);
                    var worksheet = workbook.Worksheet(sheetname);
                    foreach(int col in tile_cols){
                        foreach(int row in tile_rows){
                            // try
                            // { /
                                if (File.Exists(save_path + "/" + filename.Split("/")[filename.Split("/").Count() - 1] + "---" + sheetname + "---" + row.ToString() + "---" + col.ToString() + ".json"))
                                {
                                    // Console.WriteLine("Exists.");
                                    continue;
                                }
                                
                                int ws_first_column = worksheet.FirstColumn().ColumnNumber();
                                int ws_last_column = worksheet.LastColumn().ColumnNumber();
                                int ws_first_row = worksheet.FirstRow().RowNumber();
                                int ws_last_row = worksheet.LastRow().RowNumber();
                                
                                
                                        // Console.WriteLine("read sheet time:"+stop_watch1.ElapsedMilliseconds.ToString() + " ms");
                                get_sheet_feature(filename, sheetname, workbook, worksheet, row, col, ws_first_column, ws_last_column, ws_first_row, ws_last_row, save_path);

                                

                            // }
                            // catch
                            // {
                            //     continue;
                            // }
                        }
                    }
                }catch{
                    continue;
                }
                stop_watch.Stop();
                Console.WriteLine("all time:"+stop_watch.ElapsedMilliseconds.ToString() + " ms");
            }
                
        }

        public void batch_generate_tile_features(){
            DirectoryInfo root = new DirectoryInfo("/datadrive-2/data/fortune500_test/tile_cols/");
            FileInfo[] files=root.GetFiles();
            List<string> result = new List<string>();
            string save_path = "/datadrive-2/data/fortune500_test/tile_features/";

            foreach(var fileinfo in files){
                string origin_filename = fileinfo.Name;
                string[] splited_origin_filenames = origin_filename.Split(".json")[0].Split("---");
                string filename = splited_origin_filenames[0];
                string sheetname = splited_origin_filenames[1];

                string jsonstring1 = File.ReadAllText("/datadrive-2/data/fortune500_test/tile_cols/"+origin_filename);
                var tile_cols = JsonConvert.DeserializeObject<List<int>>(jsonstring1);
                string jsonstring2 = File.ReadAllText("/datadrive-2/data/fortune500_test/tile_rows/"+origin_filename);
                var tile_rows = JsonConvert.DeserializeObject<List<int>>(jsonstring2);

                foreach(int col in tile_cols){
                    foreach(int row in tile_rows){
                        try
                        {
                            if (File.Exists(save_path + "/" + filename.Split("/")[filename.Split("/").Count() - 1] + "---" + sheetname + "---" + row.ToString() + "---" + col.ToString() + ".json"))
                            {
                                // Console.WriteLine("Exists.");
                                continue;
                            }
                            var workbook = new XLWorkbook("/datadrive/data_fortune500/crawled_xlsx_fortune500/" + filename);
                            var worksheet = workbook.Worksheet(sheetname);
                            int ws_first_column = worksheet.FirstColumn().ColumnNumber();
                            int ws_last_column = worksheet.LastColumn().ColumnNumber();
                            int ws_first_row = worksheet.FirstRow().RowNumber();
                            int ws_last_row = worksheet.LastRow().RowNumber();
                            
                            
                                    // Console.WriteLine("read sheet time:"+stop_watch1.ElapsedMilliseconds.ToString() + " ms");
                            get_sheet_feature(filename, sheetname, workbook, worksheet, row, col, ws_first_column, ws_last_column, ws_first_row, ws_last_row, save_path);

                            

                        }
                        catch
                        {
                            continue;
                        }
                    }
                }
                
                
            }
            
        }

        public void batch_generate_refcell_features(){
            DirectoryInfo root = new DirectoryInfo("/datadrive-2/data/enron/test_refcell_position/");
            FileInfo[] files=root.GetFiles();
            List<string> result = new List<string>();
            string save_path = "/datadrive-2/data/enron/refcell_features/";
            foreach(var fileinfo in files){
                Stopwatch stop_watch = new Stopwatch();
                stop_watch.Start();
                string origin_filename = fileinfo.Name;
                Console.WriteLine("origin_filename:" + origin_filename);
                string[] splited_origin_filenames = origin_filename.Split(".json")[0].Split("---");
                string filename = splited_origin_filenames[0];
                string sheetname = splited_origin_filenames[1];

                string jsonstring1 = File.ReadAllText("/datadrive-2/data/enron/test_refcell_position/"+origin_filename);
                
                var position_list = JsonConvert.DeserializeObject<List<Dictionary<string, int>>>(jsonstring1);
                var is_all_exists = true;
                foreach(var dict in position_list){
                        int row = dict["R"];
                        int col = dict["C"];
        
                        if (!File.Exists(save_path + "/" + filename.Split("/")[filename.Split("/").Count() - 1] + "---" + sheetname + "---" + row.ToString() + "---" + col.ToString() + ".json"))
                        {
                            // Console.WriteLine("Exists.");
                            is_all_exists = false;
                            break;
                        }
                }
                if(is_all_exists){
                    Console.WriteLine("All exists.");
                    continue;
                }
                try{
                    var workbook = new XLWorkbook("/datadrive/data_fortune500/crawled_xlsx_fortune500/" + filename);
                    var worksheet = workbook.Worksheet(sheetname);
                    foreach(var dict in position_list){
                        int row = dict["R"];
                        int col = dict["C"];
        
                        if (File.Exists(save_path + "/" + filename.Split("/")[filename.Split("/").Count() - 1] + "---" + sheetname + "---" + row.ToString() + "---" + col.ToString() + ".json"))
                        {
                            // Console.WriteLine("Exists.");
                            continue;
                        }
                        
                        int ws_first_column = worksheet.FirstColumn().ColumnNumber();
                        int ws_last_column = worksheet.LastColumn().ColumnNumber();
                        int ws_first_row = worksheet.FirstRow().RowNumber();
                        int ws_last_row = worksheet.LastRow().RowNumber();
                    
                                // Console.WriteLine("read sheet time:"+stop_watch1.ElapsedMilliseconds.ToString() + " ms");
                        get_sheet_feature(filename, sheetname, workbook, worksheet, row, col, ws_first_column, ws_last_column, ws_first_row, ws_last_row, save_path);  
                    }
                }catch{
                    continue;
                }
                
                
                stop_watch.Stop();
                Console.WriteLine("all time:"+stop_watch.ElapsedMilliseconds.ToString() + " ms");
            }
            
        }


        public void get_sheet_feature_0100(string filename, string sheetname)
        {

            string[] filenames = filename.Split('/');
            string filename1 = filenames[filenames.Count() - 1];
            Console.WriteLine(filename1 + "---" + sheetname + ".json");
            try
            {
                if (!File.Exists("SheetFeatures/" + filename.Split('/')[4] + "---" + sheetname + ".json"))
                // if(!File.Exists("FeaturesDictionary/" + dvinfo.ID.ToString()+"---"+dvinfo.batch_id.ToString() + ".json") && !File.Exists("/datadrive/data/sheet_features_rgb/" + dvinfo.ID.ToString() + ".json"))
                {
                    SheetFeature sheet_feature = new SheetFeature();
                    sheet_feature.filename = filename;
                    sheet_feature.sheetname = sheetname;
                    filename = filename.Replace("/UnzipData", "");
                    var workbook = new XLWorkbook(filename);
                    Console.WriteLine("workbook:" + filename);
                    var worksheet = workbook.Worksheet(sheetname);
                    Console.WriteLine("read suc");
                    List<CellFeature> one_sheet_feature = new List<CellFeature>();

                    for (int row = 1; row <= 100; row += 1)
                    {
                        for (int column = 1; column <= 10; column += 1)
                        {
                            // Console.WriteLine("st row:"+row.ToString()+",column"+column.ToString());
                            bool exist1 = true;
                            CellFeature cell_feature = new CellFeature();
                            IXLCell cell1 = worksheet.Cell(1, 1);
                            try
                            {
                                cell1 = worksheet.Cell(row, column);
                                var value1 = cell1.Value.ToString();
                            }
                            catch
                            {
                                exist1 = false;
                            }
                            if (exist1)
                            {
                                cell_feature = get_cell_features(workbook, cell1);
                            }
                            one_sheet_feature.Add(cell_feature);
                        }
                    }
                    sheet_feature.sheetfeature = one_sheet_feature;
                    string jsonData = JsonConvert.SerializeObject(sheet_feature);

                    File.WriteAllText("SheetFeatures/" + filename.Split('/')[4] + "---" + sheetname + ".json", jsonData);

                }
            }
            catch
            {
                return;
            }

        }


    }
}