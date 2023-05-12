using System;
using ClosedXML.Excel;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Newtonsoft.Json;
using XLParser;
using System.Drawing;

namespace AnalyzeDV
{
    
    class DVSheetFeatures{
        public void saveAsJson(object need_save_content, string file_name){
            string jsonData = JsonConvert.SerializeObject(need_save_content);
            File.WriteAllText(file_name, jsonData);
        }
        public int column_id(string column_cha){
            int index=1;
            int result = 0;
            while(index<=column_cha.Count()){
                int cha_num = (int)column_cha[column_cha.Count()-index]-64;
                int sub_index=0;
                int di = 1;
                while(sub_index<index-1){
                    di*=26;
                    sub_index+=1;
                }
                result+=di*cha_num;
                index+=1;
            }
            return result;
        }

        public List<int> RangeAdress2num(string dv_range){
            List<int> result = new List<int>();
            string[] split_dv_range = dv_range.Split(':');
            string range_start_cell = split_dv_range[0];
            string range_end_cell = split_dv_range[1];
            int range_start_number_index = 0;
            foreach(var cha in range_start_cell){
                if(!Char.IsNumber(cha)){
                    range_start_number_index += 1;
                    continue;
                }
                break;
            }
            int range_end_number_index = 0;
            foreach(var cha in range_end_cell){
                if(!Char.IsNumber(cha)){
                    range_end_number_index += 1;
                    continue;
                }
                break;
            }

            string lty_c = range_start_cell.Substring(0,range_start_number_index);
            string ltx = range_start_cell.Substring(range_start_number_index);
            string rby_c = range_end_cell.Substring(0,range_end_number_index);
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
            cell_feature.content = cell.Value.ToString();
            cell_feature.content_template = get_cell_content_template(cell);
            return cell_feature;
        }


        public void batch_get_dv_sheet_features(){
            string jsonstring1 = File.ReadAllText("../analyze-dv-1/sampled_list_list.json");
            var list_dvinfos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring1);
            string jsonstring2 = File.ReadAllText("../analyze-dv-1/sampled_boundary_list.json");
            var boudnary_dvinfos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring2);
            string jsonstring3 = File.ReadAllText("data/types/custom/custom_list.json");
            var custom_dvinfos = JsonConvert.DeserializeObject<List<DVInfo>>(jsonstring3);
            List<List<DVInfo>> three_dvinfos = new List<List<DVInfo>>();
            List<string> filesheet = new List<string>();
            three_dvinfos.Add(custom_dvinfos);
            three_dvinfos.Add(list_dvinfos);
            three_dvinfos.Add(boudnary_dvinfos);
            

            foreach(var dvinfos in three_dvinfos){
                int count = 0;
                foreach(var dvinfo in dvinfos){
                // bool is_saved = false;
                    count += 1;
                    
                    // if(count==3052 || count==3053){
                    //     continue;
                    // }
                    Console.WriteLine(count.ToString()+"/"+dvinfos.Count().ToString());
                    // foreach(var fs in filesheet){
                    //     if(fs == dvinfo.FileName + "---" + dvinfo.SheetName){
                    //         is_saved=true;
                    //         break;
                    //     }
                    // }
                    // if(is_saved){
                    //     continue;
                    // }
                    string[] filenames = dvinfo.FileName.Split('/');
                    string filename1 = filenames[filenames.Count() - 1];
                    if(filename1=="4051c92706de3dc131bdc892985347e6_d3d3Lmdvb2RseS5jby5pbgkxMDQuMTguMzUuNjI=.xlsx" || filename1=="5732acf19842ca9ef8b25a3b3f881381_YWdpbGVjb25zb3J0aXVtLnBid29ya3MuY29tCTIwOC45Ni4xOC4yMzg=.xlsx"){
                        continue;
                    }
                    try{
                        Console.WriteLine("get_masked_sheet_feature......");
                        get_masked_sheet_feature(dvinfo.FileName, dvinfo.SheetName, dvinfo);
                        Console.WriteLine("get_sheet_feature......");
                        get_sheet_feature(dvinfo.FileName, dvinfo.SheetName, dvinfo);
                        Console.WriteLine("get_sheet_feature_0100......");
                        get_sheet_feature_0100(dvinfo.FileName, dvinfo.SheetName, dvinfo);
                        Console.WriteLine("get_start_column_row......");
                        get_start_column_row(dvinfo.FileName, dvinfo.SheetName, dvinfo);
                        filesheet.Add(dvinfo.FileName + "---" + dvinfo.SheetName);
                    }catch{
                        continue;
                    }
                    
                }
            }
            
        }
        public void get_start_column_row(string filename, string sheetname, CustomDVInfo dvinfo){
            string[] filenames = filename.Split('/');
            string filename1 = filenames[filenames.Count() - 1];
            Console.WriteLine(filename1 + "---" + sheetname + ".json");
            try{
                if(!File.Exists("StartRowColumn/" +dvinfo.ID.ToString()+"---"+dvinfo.batch_id.ToString() + ".json"))
                {
                    SheetFeature sheet_feature = new SheetFeature();
                    sheet_feature.filename = filename;
                    sheet_feature.sheetname = sheetname;
                    filename = filename.Replace("/UnzipData", "");
                    var workbook = new XLWorkbook(filename);
                    Console.WriteLine("workbook:"+filename);
                    var worksheet = workbook.Worksheet(sheetname);
                    Console.WriteLine("read suc");
                    List<CellFeature> one_sheet_feature = new List<CellFeature>();
                    int start_row = dvinfo.ltx-50 > 1 ? dvinfo.ltx-50 : 1;
                    int start_column = dvinfo.lty-5 > 1 ? dvinfo.lty-5 : 1;
                    List<int> start_row_column = new List<int>();
                    start_row_column.Add(start_row);
                    start_row_column.Add(start_column);
                    string jsonData1 = JsonConvert.SerializeObject(start_row_column);
                    File.WriteAllText("StartRowColumn/" +dvinfo.ID.ToString() + ".json", jsonData1);
                }else if(dvinfo.ltx==0 && dvinfo.lty==0 && dvinfo.rbx==0 && dvinfo.rby==0){
                    var range_piece = dvinfo.RangeAddress.Split(" ");
                    var range_index = new List<List<int>>();
                    foreach(string range in range_piece){
                        var start_end = range.Split(":");
                        List<int> range_start_end = new List<int>();
                        range_start_end.Add(int.Parse(start_end[0]));
                        range_start_end.Add(int.Parse(start_end[1]));
                        range_index.Add(range_start_end);
                    }
                    string jsonData1 = JsonConvert.SerializeObject(range_index);
                    File.WriteAllText("multi_StartRowColumn/" +dvinfo.ID.ToString()+"---"+dvinfo.batch_id.ToString() + ".json", jsonData1);
                }
            }catch{
                return;
            }
            
        }

        public void get_start_column_row(string filename, string sheetname, DVInfo dvinfo){
            string[] filenames = filename.Split('/');
            string filename1 = filenames[filenames.Count() - 1];
            Console.WriteLine(filename1 + "---" + sheetname + ".json");
            try{
                if(!File.Exists("StartRowColumn/" +dvinfo.ID.ToString()+"---"+dvinfo.batch_id.ToString() + ".json"))
                {
                    SheetFeature sheet_feature = new SheetFeature();
                    sheet_feature.filename = filename;
                    sheet_feature.sheetname = sheetname;
                    filename = filename.Replace("/UnzipData", "");
                    var workbook = new XLWorkbook(filename);
                    Console.WriteLine("workbook:"+filename);
                    var worksheet = workbook.Worksheet(sheetname);
                    Console.WriteLine("read suc");
                    List<CellFeature> one_sheet_feature = new List<CellFeature>();
                    List<int> range_index = RangeAdress2num(dvinfo.RangeAddress);
                    int ltx = range_index[0];
                    int lty = range_index[1];
                    int rbx = range_index[2];
                    int rby = range_index[3];
                    int start_row = ltx-50 > 1 ? ltx-50 : 1;
                    int start_column = lty-5 > 1 ? lty-5 : 1;
                    List<int> start_row_column = new List<int>();
                    start_row_column.Add(start_row);
                    start_row_column.Add(start_column);
                    string jsonData1 = JsonConvert.SerializeObject(start_row_column);
                    File.WriteAllText("StartRowColumn/" +dvinfo.ID.ToString()+"---"+dvinfo.batch_id.ToString() + ".json", jsonData1);
                }
            }catch{
                return;
            }
            
        }
        public void get_masked_sheet_feature(string filename, string sheetname, CustomDVInfo dvinfo){
    
            string[] filenames = filename.Split('/');
            string filename1 = filenames[filenames.Count() - 1];
            Console.WriteLine(filename1 + "---" + sheetname + ".json");
            try{
                if(!File.Exists("DVFeaturesDictionary/" + dvinfo.ID.ToString()+"---"+dvinfo.batch_id.ToString() + ".json"))
                {
                    SheetFeature sheet_feature = new SheetFeature();
                    sheet_feature.filename = filename;
                    sheet_feature.sheetname = sheetname;
                    filename = filename.Replace("/UnzipData", "");
                    var workbook = new XLWorkbook(filename);
                    Console.WriteLine("workbook:"+filename);
                    var worksheet = workbook.Worksheet(sheetname);
                    Console.WriteLine("read suc");
                    List<CellFeature> one_sheet_feature = new List<CellFeature>();
                    int start_row = dvinfo.ltx-50 > 1 ? dvinfo.ltx-50 : 1;
                    int start_column = dvinfo.lty-5 > 1 ? dvinfo.lty-5 : 1;
                    List<int> start_row_column = new List<int>();
                    start_row_column.Add(start_row);
                    start_row_column.Add(start_column);

                    for (int row = start_row; row <= start_row + 99; row += 1)
                    {
                        for (int column = start_column; column <= start_column + 9; column += 1)
                        {
                            Console.WriteLine("dv row:"+row.ToString()+",column"+column.ToString());
                            CellFeature cell_feature = new CellFeature();
                            if(row>=dvinfo.ltx && row <= dvinfo.rbx && column>=dvinfo.lty && column <= dvinfo.rby){
                                bool exist1 = true;
                                
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
                            }
                            one_sheet_feature.Add(cell_feature);
                        }
                    }
                    sheet_feature.sheetfeature = one_sheet_feature;
                    string jsonData = JsonConvert.SerializeObject(sheet_feature);

                    File.WriteAllText("DVFeaturesDictionary/" + dvinfo.ID.ToString()+"---"+dvinfo.batch_id.ToString() + ".json", jsonData);
                    string jsonData1 = JsonConvert.SerializeObject(start_row_column);
                    File.WriteAllText("StartRowColumn/" +dvinfo.ID.ToString()+"---"+dvinfo.batch_id.ToString() + ".json", jsonData1);
                }
            }catch{
                return;
            }
            
        }

        public void get_masked_sheet_feature(string filename, string sheetname, DVInfo dvinfo){
    
            string[] filenames = filename.Split('/');
            string filename1 = filenames[filenames.Count() - 1];
            Console.WriteLine(filename1 + "---" + sheetname + ".json");
            try{
                if(!File.Exists("DVFeaturesDictionary/" + dvinfo.ID.ToString()+"---"+dvinfo.batch_id.ToString() + ".json"))
                {
                    // Console.WriteLine("not exist");
                    SheetFeature sheet_feature = new SheetFeature();
                    sheet_feature.filename = filename;
                    sheet_feature.sheetname = sheetname;
                    filename = filename.Replace("/UnzipData", "");
                    // Console.WriteLine("before load workbook");
                    var workbook = new XLWorkbook(filename);
                    Console.WriteLine("workbook:"+filename);
                    var worksheet = workbook.Worksheet(sheetname);
                    Console.WriteLine("read suc");
                    List<CellFeature> one_sheet_feature = new List<CellFeature>();
                    List<int> range_index = RangeAdress2num(dvinfo.RangeAddress);
                    int ltx = range_index[0];
                    int lty = range_index[1];
                    int rbx = range_index[2];
                    int rby = range_index[3];
                    int start_row = ltx-50 > 1 ? ltx-50 : 1;
                    int start_column = lty-5 > 1 ? lty-5 : 1;
                    List<int> start_row_column = new List<int>();
                    start_row_column.Add(start_row);
                    start_row_column.Add(start_column);

                    for (int row = start_row; row <= start_row + 99; row += 1)
                    {
                        for (int column = start_column; column <= start_column + 9; column += 1)
                        {
                            Console.WriteLine("dv row:"+row.ToString()+",column"+column.ToString());
                            CellFeature cell_feature = new CellFeature();
                            if(row>=ltx && row <= rbx && column>=lty && column <= rby){
                                bool exist1 = true;
                                
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
                            }
                            one_sheet_feature.Add(cell_feature);
                        }
                    }
                    sheet_feature.sheetfeature = one_sheet_feature;
                    string jsonData = JsonConvert.SerializeObject(sheet_feature);

                    File.WriteAllText("DVFeaturesDictionary/" + dvinfo.ID.ToString()+"---"+dvinfo.batch_id.ToString() + ".json", jsonData);
                    string jsonData1 = JsonConvert.SerializeObject(start_row_column);
                    File.WriteAllText("StartRowColumn/" +dvinfo.ID.ToString()+"---"+dvinfo.batch_id.ToString() + ".json", jsonData1);
                // }else{
                    // Console.WriteLine("exist");
                }
            }catch{
                // Console.WriteLine("error");
                return;
            }
            
        }

        public void get_sheet_feature(string filename, string sheetname, CustomDVInfo dvinfo){
    
            string[] filenames = filename.Split('/');
            string filename1 = filenames[filenames.Count() - 1];
            Console.WriteLine(filename1 + "---" + sheetname + ".json");
            try{
                if(!File.Exists("NMDVFeaturesDictionary/" + dvinfo.ID.ToString()+"---"+dvinfo.batch_id.ToString() + ".json"))
                {
                    SheetFeature sheet_feature = new SheetFeature();
                    sheet_feature.filename = filename;
                    sheet_feature.sheetname = sheetname;
                    filename = filename.Replace("/UnzipData", "");
                    var workbook = new XLWorkbook(filename);
                    Console.WriteLine("workbook:"+filename);
                    var worksheet = workbook.Worksheet(sheetname);
                    Console.WriteLine("read suc");
                    List<CellFeature> one_sheet_feature = new List<CellFeature>();
                    int start_row = dvinfo.ltx-50 > 1 ? dvinfo.ltx-50 : 1;
                    int start_column = dvinfo.lty-5 > 1 ? dvinfo.lty-5 : 1;
                    List<int> start_row_column = new List<int>();
                    start_row_column.Add(start_row);
                    start_row_column.Add(start_column);

                    for (int row = start_row; row <= start_row + 99; row += 1)
                    {
                        for (int column = start_column; column <= start_column + 9; column += 1)
                        {
                            Console.WriteLine("dv row:"+row.ToString()+",column"+column.ToString());
                            CellFeature cell_feature = new CellFeature();
                        // if(row>=dvinfo.ltx && row <= dvinfo.rbx && column>=dvinfo.lty && column <= dvinfo.rby){
                            bool exist1 = true;
                            
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
                            // }
                            one_sheet_feature.Add(cell_feature);
                        }
                    }
                    sheet_feature.sheetfeature = one_sheet_feature;
                    string jsonData = JsonConvert.SerializeObject(sheet_feature);

                    File.WriteAllText("NMDVFeaturesDictionary/" + dvinfo.ID.ToString()+"---"+dvinfo.batch_id.ToString() + ".json", jsonData);
                    string jsonData1 = JsonConvert.SerializeObject(start_row_column);
                    File.WriteAllText("StartRowColumn/" +dvinfo.ID.ToString()+"---"+dvinfo.batch_id.ToString() + ".json", jsonData1);
                }
            }catch{
                return;
            }
            
        }

        public void get_sheet_feature(string filename, string sheetname, DVInfo dvinfo){
    
            string[] filenames = filename.Split('/');
            string filename1 = filenames[filenames.Count() - 1];
            Console.WriteLine(filename1 + "---" + sheetname + ".json");
            try{
                if(!File.Exists("NMDVFeaturesDictionary/" + dvinfo.ID.ToString()+"---"+dvinfo.batch_id.ToString() + ".json"))
                {
                    SheetFeature sheet_feature = new SheetFeature();
                    sheet_feature.filename = filename;
                    sheet_feature.sheetname = sheetname;
                    filename = filename.Replace("/UnzipData", "");
                    var workbook = new XLWorkbook(filename);
                    Console.WriteLine("workbook:"+filename);
                    var worksheet = workbook.Worksheet(sheetname);
                    Console.WriteLine("read suc");
                    List<CellFeature> one_sheet_feature = new List<CellFeature>();
                    List<int> range_index = RangeAdress2num(dvinfo.RangeAddress);
                    int ltx = range_index[0];
                    int lty = range_index[1];
                    int rbx = range_index[2];
                    int rby = range_index[3];
                    int start_row = ltx-50 > 1 ? ltx-50 : 1;
                    int start_column = lty-5 > 1 ? lty-5 : 1;
                    List<int> start_row_column = new List<int>();
                    start_row_column.Add(start_row);
                    start_row_column.Add(start_column);

                    for (int row = start_row; row <= start_row + 99; row += 1)
                    {
                        for (int column = start_column; column <= start_column + 9; column += 1)
                        {
                            Console.WriteLine("dv row:"+row.ToString()+",column"+column.ToString());
                            CellFeature cell_feature = new CellFeature();
                        // if(row>=dvinfo.ltx && row <= dvinfo.rbx && column>=dvinfo.lty && column <= dvinfo.rby){
                            bool exist1 = true;
                            
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
                            // }
                            one_sheet_feature.Add(cell_feature);
                        }
                    }
                    sheet_feature.sheetfeature = one_sheet_feature;
                    string jsonData = JsonConvert.SerializeObject(sheet_feature);

                    File.WriteAllText("NMDVFeaturesDictionary/" + dvinfo.ID.ToString()+"---"+dvinfo.batch_id.ToString() + ".json", jsonData);
                    string jsonData1 = JsonConvert.SerializeObject(start_row_column);
                    File.WriteAllText("StartRowColumn/" +dvinfo.ID.ToString()+"---"+dvinfo.batch_id.ToString() + ".json", jsonData1);
                }
            }catch{
                return;
            }
            
        }

        public void get_sheet_feature_0100(string filename, string sheetname, CustomDVInfo dvinfo){
    
            string[] filenames = filename.Split('/');
            string filename1 = filenames[filenames.Count() - 1];
            Console.WriteLine(filename1 + "---" + sheetname + ".json");
            try{
                if(!File.Exists("FeaturesDictionary/" + dvinfo.ID.ToString()+"---"+dvinfo.batch_id.ToString() + ".json") && !File.Exists("/datadrive/data/sheet_features_rgb/" + dvinfo.ID.ToString() + ".json"))
                {
                    SheetFeature sheet_feature = new SheetFeature();
                    sheet_feature.filename = filename;
                    sheet_feature.sheetname = sheetname;
                    filename = filename.Replace("/UnzipData", "");
                    var workbook = new XLWorkbook(filename);
                    Console.WriteLine("workbook:"+filename);
                    var worksheet = workbook.Worksheet(sheetname);
                    Console.WriteLine("read suc");
                    List<CellFeature> one_sheet_feature = new List<CellFeature>();
                    int start_row = dvinfo.ltx-50 > 0 ? dvinfo.ltx-50 : 0;
                    int start_column = dvinfo.lty-5 > 0 ? dvinfo.lty-5 : 0;

                    for (int row = 1; row <= 100; row += 1)
                    {
                        for (int column = 1; column <= 10; column += 1)
                        {
                            Console.WriteLine("st row:"+row.ToString()+",column"+column.ToString());
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

                    File.WriteAllText("FeaturesDictionary/" + dvinfo.ID.ToString()+"---"+dvinfo.batch_id.ToString() + ".json", jsonData);
            
                }
            }catch{
                return;
            }
            
        }

        public void get_sheet_feature_0100(string filename, string sheetname, DVInfo dvinfo){
    
            string[] filenames = filename.Split('/');
            string filename1 = filenames[filenames.Count() - 1];
            Console.WriteLine(filename1 + "---" + sheetname + ".json");
            try{
                if(!File.Exists("FeaturesDictionary/" + dvinfo.ID.ToString()+"---"+dvinfo.batch_id.ToString() + ".json") && !File.Exists("/datadrive/data/sheet_features_rgb/" + dvinfo.ID.ToString() + ".json"))
                {
                    SheetFeature sheet_feature = new SheetFeature();
                    sheet_feature.filename = filename;
                    sheet_feature.sheetname = sheetname;
                    filename = filename.Replace("/UnzipData", "");
                    var workbook = new XLWorkbook(filename);
                    Console.WriteLine("workbook:"+filename);
                    var worksheet = workbook.Worksheet(sheetname);
                    Console.WriteLine("read suc");
                    List<CellFeature> one_sheet_feature = new List<CellFeature>();

                    for (int row = 1; row <= 100; row += 1)
                    {
                        for (int column = 1; column <= 10; column += 1)
                        {
                            Console.WriteLine("st row:"+row.ToString()+",column"+column.ToString());
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

                    File.WriteAllText("FeaturesDictionary/" + dvinfo.ID.ToString()+"---"+dvinfo.batch_id.ToString() + ".json", jsonData);
            
                }
            }catch{
                return;
            }
            
        }
    }
}