using ClosedXML.Excel;
using System.Collections.Generic;

namespace AnalyzeDV
{
    class CellFormula
    {
        public int row;
        public int column;
        public string formulaA1;
        public string formulaR1C1;
        public int formulaReferenceFC;
        public int formulaReferenceFR;
        public int formulaReferenceLC;
        public int formulaReferenceLR;
    }
    class SimpleCellFormula{
        public int column;
        public int row;
        public string formulaA1;
        public string formulaR1C1;
    }
    class Formula
    {
        public int fr;
        public int fc;
        public int lr;
        public int lc;
        public string r1c1;
        public List<SimpleCellFormula> formulas;
    }
    class SimpleFormula
    {
        public int fc;
        public int fr;
        public int lc;
        public int lr;
    }
}
