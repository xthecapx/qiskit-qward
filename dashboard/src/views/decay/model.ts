import { algorithmName } from "../../lib/domain/catalog";
import { palette, type EChartsCoreOption } from "../../lib/charts/echarts";

type Point = [number, number, number, number, number];
export interface Period {
  slopeA: number; slopeB: number; gap: number; rho: number | null; points: Point[];
}
export interface Cell {
  a: string; b: string; role: string; gap: number | null; grouped: boolean | null;
  periods: Record<string, Period>; overlap: Record<string, number>;
}
export interface Snapshot {
  schemaVersion: number; backend: string; jobs: number; algorithms: string[];
  tolerance: number; minPeriods: number; minLevels: number;
  panels: Record<string, Record<string, Cell[]>>;
}
export const resources: Record<string,string> = {
  "a.num_qubits":"Total logical qubits", "a.depth":"Logical depth", "a.size":"Logical operations",
  "a.ComplexityMetrics.gate_based_metrics.two_qubit_count":"Logical two-qubit gates",
  "a.StructuralMetrics.volume":"Structural volume", "a.BehavioralMetrics.program_communication":"Program communication",
  "a.quantum_specific.entanglement_ratio":"Entanglement ratio", "b.depth":"Compiled depth",
  "b.two_qubit_depth":"Compiled two-qubit depth", "b.two_qubit_gates":"Compiled two-qubit gates",
};
export const targets: Record<string,string> = {success:"Success rate", normalized:"Chance-adjusted, ideal-normalized success", hellinger_fidelity:"Hellinger fidelity", tvd_similarity:"TVD similarity", dsr:"Differential success rate"};
export const roles: Record<string,string> = {fourier:"Fourier family", oracle:"Oracle family", entanglement:"Entanglement family", composition:"Fourier composition", mirror_protocol:"Mirror validation comparison", control:"Random-volumetric control", other:"Cross-family comparison"};
export const short = (a:string) => ({"BERNSTEIN-VAZIRANI":"BV", "PHASE-ESTIMATION":"PE", "RANDOM-VOLUMETRIC":"RV"}[a] ?? algorithmName(a));
export const key = (c:Cell) => c.a + "/" + c.b;
export const fixed = (v:number|null) => v == null ? "—" : v.toFixed(2);
// The row and column use the same observed resource range in each period.
export function extraLoss(period: Period, reversed = false): number {
  const span = period.points[period.points.length - 1][0] - period.points[0][0];
  return (reversed ? -1 : 1) * 100 * (period.slopeA - period.slopeB) * span;
}
export function medianLoss(cell: Cell): number | null {
  const values = Object.values(cell.periods).map(p => extraLoss(p)).sort((a, b) => a - b);
  if (!values.length) return null;
  const mid = Math.floor(values.length / 2);
  return values.length % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
}
export function signed(value: number | null, digits = 2): string {
  if (value === null) return "—";
  const rounded = Number(value.toFixed(digits));
  return (rounded > 0 ? "+" : "") + rounded.toFixed(digits);
}
export function bounds(periods:Period[]) {
  const ys=periods.flatMap(p=>p.points.flatMap(t=>t.slice(1).map(v=>100*v)));
  const xs=periods.flatMap(p=>p.points.map(t=>t[0]));
  return {min: Math.floor(Math.min(0,...ys)/10)*10, max: Math.ceil(Math.max(100,...ys)/10)*10,
    xmin:Math.min(...xs), xmax:Math.max(...xs)};
}
export function chart(p:Period,c:Cell,b:ReturnType<typeof bounds>,x:string,y:string):EChartsCoreOption {
  return {
    grid:{top:20,left:48,right:15,bottom:49},
    xAxis:{type:"value",min:b.xmin,max:b.xmax,splitNumber:4,name:x,nameLocation:"middle",nameGap:29,nameTextStyle:{fontSize:10}},
    yAxis:{type:"value",min:b.min,max:b.max,splitNumber:4,axisLabel:{fontSize:10}},
    tooltip:{trigger:"item",confine:true},
    series:([c.a,c.b]).flatMap((name,i) => {
      const color=i===0?palette.teal:palette.accent;
      return [
        {type:"line" as const,name:algorithmName(name)+" fit",showSymbol:false,lineStyle:{color,width:2.5,type:i===0?"solid" as const:"dashed" as const},silent:true,data:p.points.map(t=>[t[0],100*t[3+i]])},
        {type:"scatter" as const,name:algorithmName(name),symbol:i===0?"circle":"diamond",symbolSize:9,itemStyle:{color},dimensions:[x,y],data:p.points.map(t=>[t[0],+(100*t[1+i]).toFixed(2)])}
      ];
    })
  };
}
