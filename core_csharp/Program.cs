using System;
using ClosedXML.Excel;
using System.Threading;

namespace AnalyzeDV
{ 
    class test{
        public void test1(){
            var worksheet1 = new XLWorkbook("../../data/UnzipData/042/6f1f29f9-0b45-4134-8a4c-425fb68611d3_aHR0cHM6Ly9kcmFnb25saW5rLmRyZXhlbC5lZHUvb3JnYW5pemF0aW9uL2luZGlhbi11bmRlcmdyYWR1YXRlLXN0dWRlbnRzLWFzc29jaWF0aW9uL2RvY3VtZW50cy92aWV3LzU4NDc1Mw==.xlsx").Worksheet("General Information");
            var worksheet2 = new XLWorkbook("../../data/UnzipData/052/25032724-fecd-458c-8294-25623df212da_aHR0cHM6Ly9kcmFnb25saW5rLmRyZXhlbC5lZHUvb3JnYW5pemF0aW9uL2RyZXhlbC1tYWdpYy10aGUtZ2F0aGVyaW5nLWNsdWIvZG9jdW1lbnRzL3ZpZXcvNTI1MjQ5.xlsx").Worksheet("General Information");
            foreach(var dv in worksheet1.DataValidations){
                Console.WriteLine(dv.AllowedValues.ToString());
                Console.WriteLine(dv.Ranges.ToString());
            }
            Console.WriteLine("###################################");
            foreach(var dv in worksheet2.DataValidations){
                Console.WriteLine(dv.AllowedValues.ToString());
                Console.WriteLine(dv.Ranges.ToString());
            }
        }
    }
    class Program
    {
        
        
        public static void test_formula(){
            using (var workbook = new XLWorkbook("test111.xlsx")){
                foreach(var sheet in workbook.Worksheets){
                    if(sheet.Name != "Sheet1"){
                        continue;
                    }
                    for(var row=1; row<20; row++){
                        for(var column=1; column<3; column++){
                            Console.WriteLine("xxxxxxxxxxxx");
                            Console.WriteLine("row:"+row.ToString()+",column:"+column.ToString());
                            var cell = sheet.Cell(row, column);
                            Console.WriteLine(cell.FormulaA1.Length);
                            Console.WriteLine(cell.FormulaA1);
                            Console.WriteLine(cell.FormulaR1C1);
                            Console.WriteLine(cell.FormulaReference);
                        }
                    }
                }
            }
        }
        static void Main(string[] args)
        {
            // RefShift rs = new RefShift();
            // rs.batch_get_tile_point();
            
            // FormulaSheetFeatures fsf = new FormulaSheetFeatures();
            // fsf.batch_generate_tile_features();
            // fsf.batch_generate_refcell_features();
            // fsf.batch_generate_deep_tile_features();
            // fsf.batch_get_dv_sheet_features();
            // fsf.batch_get_sheet_features();
            // StartProcess1 sp = new StartProcess1();
            // sp.Do();

            // test_formula();
            // DVSheetFeatures dvsf = new DVSheetFeatures();
            // dvsf.batch_get_dv_sheet_features();
            // FormulaSheetFeatures fsf = new FormulaSheetFeatures();
            // fsf.batch_get_neighbors_features();
            // fsf.batch_get_another_sheet_features();
            // dvsf.batch_get_dv_sheet_features();
            // TestProcess test_process = new TestProcess();
            // test_process.test_start();

            // ThreadStart childref1_0 = new ThreadStart(CallToChildThread_1_0);
            // Console.WriteLine("In Main: Creating the Child thread 1");
            // Thread childThread1_0 = new Thread(childref1_0);
            // // childThread1_0.IsBackground = true;
            // childThread1_0.Start();

            // ThreadStart childref1_1 = new ThreadStart(CallToChildThread_1_1);
            // Console.WriteLine("In Main: Creating the Child thread 1");
            // Thread childThread1_1 = new Thread(childref1_1);
            // // childThread1_1.IsBackground = true;
            // childThread1_1.Start();

            

            // ThreadStart childref1_2 = new ThreadStart(CallToChildThread_1_2);
            // Console.WriteLine("In Main: Creating the Child thread 1");
            // Thread childThread1_2 = new Thread(childref1_2);
            // // childThread1_2.IsBackground = true;
            // childThread1_2.Start();

            // ThreadStart childref1_3 = new ThreadStart(CallToChildThread_1_3);
            // Console.WriteLine("In Main: Creating the Child thread 1");
            // Thread childThread1_3 = new Thread(childref1_3);
            // // childThread1_3.IsBackground = true;
            // childThread1_3.Start();

            // ThreadStart childref1_4 = new ThreadStart(CallToChildThread_1_4);
            // Console.WriteLine("In Main: Creating the Child thread 1");
            // Thread childThread1_4 = new Thread(childref1_4);
            // // childThread1_3.IsBackground = true;
            // childThread1_4.Start();

            // ThreadStart childref1_5 = new ThreadStart(CallToChildThread_1_5);
            // Console.WriteLine("In Main: Creating the Child thread 1");
            // Thread childThread1_5 = new Thread(childref1_5);
            // // childThread1_3.IsBackground = true;
            // childThread1_5.Start();
            
            // ThreadStart childref0 = new ThreadStart(CallToChildThread_0);
            // Console.WriteLine("In Main: Creating the Child thread 0");
            // Thread childThread0 = new Thread(childref0);
            // // childThread0.IsBackground = true;
            // childThread0.Start();
            // childThread0.Join();
            // TestFormula tf = new TestFormula();
            // tf.get_training_refcell();
            // test_formula.test_parse_refence();
            // analyzer.analyze_formula_fortune500();
            // analyzer.analyze_formula_top10domain();
            // analyzer.analyze_formula();
            // analyzer.cluster_boundary();
            // analyzer.get_all_custom_sheets();
            // analyzer.generate_sampled_files();
            // analyzer.analyze_training_formula();
            // analyzer.cluster_list();
            // analyzer.getMetaInfo();
            // analyzer.countAllSheet();
            // analyzer.countAllSheet();
            // test test = new test();
            // test.test1();
            // SheetSim sheet_sim = new SheetSim();
            // sheet_sim.get_training_sheet_name();
            // sheet_sim.devide_by_file_template();
            // sheet_sim.batch_get_sheet_features();
            // analyzer.getMetaInfo();
            TestFormula test_formula = new TestFormula();
            // test_formula.get_all_refcell("R[-1]C*12/7");
            // test_formula.batch_get_all_refcell();

            // test_formula.save_all_template(false);
            test_formula.save_all_formula_template(true);
            // test_formula.id2sheetnames();
            // test_formula.analyze_differ();
            // Excuter test_formula = new Excuter();
            // test_formula.test_evaluate();
            // Analyzer analyzer = new Analyzer();
            // analyzer.sheetfile();
            // SheetSim sheetsim = new SheetSim();
            // sheetsim.get_same_sheet_in_metric();
            // sheetsim.batch_get_sheet_features();
            // sheetsim.check_not_same();
            // sheetsim.devide_by_file_template();
            // sheet_sim.get_sheet_number();
            // sheetsim.get_training_sheet_name();
            // sheetsim.get_training_sheet_name();

            // test_formula.save_all_template(false);
            
            // analyzer.getMetaInfo();
            // analyzer.getCustomMetaInfo();
            // analyzer.getBoundary(XLAllowedValues.Date);
            // analyzer.getBoundary(XLAllowedValues.Time);
            // analyzer.getBoundary(XLAllowedValues.TextLength);
            // analyzer.getBoundary(XLAllowedValues.Decimal);
            // analyzer.getBoundary(XLAllowedValues.WholeNumber);
            // analyzer.getFunctionCount();
            // analyzer.getFunctionFiles("AND");


            // analyzer.filtCustomFunction();
            // analyzer.filtList();
            // Merger merger= new Merger();
            // merger.filt_list("../share/dvinfoWithRef2.json", "new_filter_list_batch_2.json");
            // merger.get_same_dict(XLAllowedValues.List, "new_filter_list_batch_2.json", "new_same_dict_batch_2.json");
            // merger.get_continous(XLAllowedValues.List,"new_same_dict_batch_2.json", "new_continous_batch_2.json");
            // analyzer.count_list_type();
            // analyzer.count_global_refer_list();
            // analyzer.sample_fifty_dvs();
            // analyzer.count_filter_list_type();
            // merger.test_type_1("../share/dvinfoWithRef.json");
            // for(var i=2; i<=3; i++){
            //     if(i == 0){
            //         merger.recheck("../share/dvinfoWithRef.json", "dvinfoWithRef0.json");
            //     }else{
            //         merger.recheck("../share/dvinfoWithRef"+i.ToString()+".json", "dvinfoWithRef0"+i.ToString()+".json");
            //     }
            // }
            // merger.recheck("../share/dvinfoWithRef.json", "dvinfoWithRef0.json", 20);
            // merger.get_same_dict();

            // RemoveFormulas rf = new RemoveFormulas();
            // rf.remove_formulas();
            // rf.print_formulas();
        }
    }
}
