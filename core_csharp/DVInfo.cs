using System;
using ClosedXML.Excel;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text;
using Newtonsoft.Json;

namespace AnalyzeDV
{
    class ReferItem{
        public object Value;
        public XLDataType DataType;
    }
    class Refer{
        public int Type;
        public List<ReferItem> List;

    }
    class CustomDVInfo{
        public int ID;
        public XLAllowedValues Type;
        public XLOperator Operator;
        public string Value;
        public string MinValue;
        public string MaxValue;
        public string InputTitle;
        public string InputMessage;
        public string ErrorTitle;
        public string ErrorMessage;
        public XLErrorStyle ErrorStyle;
        public string RangeAddress;
        public int Height;
        public int Width;
        public string FileName;
        public string SheetName;
        public List<Tuple> content;
        public List<Tuple> header;
        // public Dictionary<string, object> refers;
        public Refer refers;
        public int batch_id;
        public int ltx;
        public int lty;
        public int rbx;
        public int rby;
        public List<Shift> shift;
    }
    class DVInfo{
        public int ID;
        public XLAllowedValues Type;
        public XLOperator Operator;
        public string Value;
        public string MinValue;
        public string MaxValue;
        public string InputTitle;
        public string InputMessage;
        public string ErrorTitle;
        public string ErrorMessage;
        public XLErrorStyle ErrorStyle;
        public string RangeAddress;
        public int Height;
        public int Width;
        public string FileName;
        public string SheetName;
        public List<Tuple> content;
        public List<Tuple> header;
        // public Dictionary<string, object> refers;
        public Refer refers;
        public int batch_id;
    };
}