using System;
using ClosedXML.Excel;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Newtonsoft.Json;
using XLParser;
using System.Diagnostics;
using System.ComponentModel;

namespace AnalyzeDV
{
    
    class TestProcess{
        public void test_start(){
            string jsonstring1 = File.ReadAllText("../analyze-dv-1/100000_positive_need_feature.json");
            var positive_sheet_2_file = JsonConvert.DeserializeObject<Dictionary<string, List<string>>>(jsonstring1);
            
            Dictionary<string, List<string>> positive_file2sheets = new Dictionary<string, List<string>>();
            foreach(var sheetname in positive_sheet_2_file.Keys){
                foreach(var file in positive_sheet_2_file[sheetname]){
                    if(!positive_file2sheets.ContainsKey(file)){
                        List<string> sheetname_list = new List<string>();
                        positive_file2sheets.Add(file, sheetname_list);
                    }
                    positive_file2sheets[file].Add(sheetname);
                }
            }

            foreach(var filename in positive_file2sheets.Keys){
                foreach(var sheetname in positive_file2sheets[filename]){
                    
                    
                    var p = new Process();
                    p.StartInfo = new ProcessStartInfo
                    {
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        FileName = "../ExtractFeature/ExtractFeature/bin/Debug/net5.0/win10-x64/ExtractFeature.exe",
                        Arguments = $"{filename} {sheetname}"
                    };
                    var list = new List<string>();
                    var downloadtime = new List<string>();
                    var extracttime = new List<string>();
                    p.OutputDataReceived += (sender, e) =>
                    {
                        if (e.Data != null)
                        {
                            if (e.Data.StartsWith("Time"))
                            {
                                var time = e.Data.Split(':')[1].Trim('$');
                                if (e.Data.Contains("download"))
                                {
                                    downloadtime.Add(time);
                                }
                                if (e.Data.Contains("extract"))
                                {
                                    extracttime.Add(time);
                                }
                            }
                            else
                            {
                                if (e.Data.StartsWith("ERR:"))
                                {
                                    extracttime.Add("EEE");
                                }
                                list.Add(e.Data);
                            }
                        }
                    };
                    p.ErrorDataReceived += (sender, e) =>
                    {
                        if (e.Data != null)
                        {
                            Console.WriteLine("A");
                        }
                    };
                    p.Start();
                    Stopwatch watch = new Stopwatch();
                    p.BeginOutputReadLine();
                    if (!p.WaitForExit(180000))
                    {
                        list.Add($"ERR-TIME:\t{filename}\t{sheetname}\t{list.Count}");
                        // Console.WriteLine($" Time out at {list.Count + start}");
                        // result.AddRange(list);
                        p.Kill();
                        var maxlen = downloadtime.Concat(extracttime).Max(s => s.Length);
                        Console.WriteLine("\tD:" + string.Join(',', downloadtime.Select(v => v.PadLeft(maxlen, ' '))));
                        Console.WriteLine("\tE:" + string.Join(',', extracttime.Select(v => v.PadLeft(maxlen, ' '))));
                        // Run6(idx, batch, result, result.Count, filetime);
                    }
                    else
                    {
                        var maxlen = downloadtime.Concat(extracttime).Max(s => s.Length);
                        Console.WriteLine($" Done T = {watch.ElapsedMilliseconds}, E# = {list.Count(l => l.StartsWith("ERR"))}");
                        Console.WriteLine("\tD:" + string.Join(',', downloadtime.Select(v => v.PadLeft(maxlen, ' '))));
                        Console.WriteLine("\tE:" + string.Join(',', extracttime.Select(v => v.PadLeft(maxlen, ' '))));
                        // result.AddRange(list);
                        watch.Stop();
                        // Run6(idx, batch, result, result.Count, filetime);
                    }
                    break;
                }
                break;
            }
        }

    }
}