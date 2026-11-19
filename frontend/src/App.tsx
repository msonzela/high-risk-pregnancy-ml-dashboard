import { useEffect, useState } from 'react';
import { Activity, BarChart3, BrainCircuit, ClipboardCheck, FlaskConical, HeartPulse, Menu, ShieldAlert } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import './index.css';

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
const WARNING = 'Synthetic research demonstration only. Model scores are not clinically validated probabilities and must not be used for diagnosis, triage, treatment, or patient care.';
type Json = Record<string, any>;
type Page = 'Overview' | 'Model Performance' | 'Explainable AI' | 'Model Validation' | 'Prediction Simulator' | 'Research Limitations';
const pages: {name: Page; icon: any}[] = [
  {name:'Overview',icon:Activity},{name:'Model Performance',icon:BarChart3},{name:'Explainable AI',icon:BrainCircuit},
  {name:'Model Validation',icon:ClipboardCheck},{name:'Prediction Simulator',icon:FlaskConical},{name:'Research Limitations',icon:ShieldAlert}
];
const initial = { maternal_age:28, parity:1, gestational_age_weeks:32, anc_visits:4, bmi:25,
  previous_hypertension:0, previous_adverse_outcome:0, systolic_bp:120, diastolic_bp:80,
  haemoglobin_g_dl:11.5, region:'', maternal_age_group:'', bmi_group:'' };
const numericFields: Array<keyof typeof initial> = ['maternal_age','parity','gestational_age_weeks','anc_visits','bmi','previous_hypertension','previous_adverse_outcome','systolic_bp','diastolic_bp','haemoglobin_g_dl'];
const pretty = (s:string) => s.replaceAll('_',' ').replace(/\b\w/g, c=>c.toUpperCase());
const number = (n:any, digits=3) => typeof n === 'number' && Number.isFinite(n) ? n.toFixed(digits) : 'Not available';
const percent = (n:any) => typeof n === 'number' && Number.isFinite(n) ? `${(n*100).toFixed(1)}%` : 'Not available';
const pick = (o:Json | null, ...keys:string[]) => { for(const k of keys) { const v = o?.[k]; if(v !== undefined && v !== null) return v; } return undefined; };
async function get(path:string):Promise<Json | null> { const res=await fetch(`${API}${path}`); if(res.status===404) return null; if(!res.ok) throw new Error(`${path}: HTTP ${res.status}`); return res.json(); }
function Metric({label,value,detail}:{label:string,value:string,detail?:string}) { return <div className="metric"><span>{label}</span><strong>{value}</strong>{detail&&<small>{detail}</small>}</div>; }
function Panel({title,children}:{title:string,children:React.ReactNode}) {return <section className="panel"><h2>{title}</h2>{children}</section>}
export default function App() {
 const [page,setPage]=useState<Page>('Overview');
 const [menu,setMenu]=useState(false);
 const [metrics,setMetrics]=useState<Json|null>(null);
 const [validation,setValidation]=useState<Json|null>(null);
 const [health,setHealth]=useState<Json|null>(null);
 const [shap,setShap]=useState<Json|null>(null);
 const [shapError,setShapError]=useState('');
 const [loading,setLoading]=useState(true);
 const [error,setError]=useState('');
 const [form,setForm]=useState(initial);
 const [prediction,setPrediction]=useState<Json|null>(null);
 const [predictError,setPredictError]=useState('');
 const [predicting,setPredicting]=useState(false);
 useEffect(()=>{ let live=true; Promise.allSettled([get('/api/model/metrics'),get('/api/model/validation'),get('/api/health'),get('/api/model/shap')]).then(results=>{
   if(!live)return;
   if(results[0].status==='fulfilled')setMetrics(results[0].value);
   if(results[1].status==='fulfilled')setValidation(results[1].value);
   if(results[2].status==='fulfilled')setHealth(results[2].value);
   if(results[3].status==='fulfilled') {
     setShap(results[3].value);
     if(results[3].value===null) setShapError('SHAP endpoint not found (HTTP 404). Check backend/main.py and reports/shap_feature_importance.csv.');
   } else {
     setShapError(results[3].reason instanceof Error ? results[3].reason.message : 'Unable to load SHAP results.');
   }
   if(results.every(r=>r.status==='rejected'))setError('Backend is unreachable. Start FastAPI on port 8000.');
   setLoading(false);
 }); return()=>{live=false}; },[]);
 const evaluation=metrics?.evaluation ?? metrics ?? {};
 const chosenName=pick(metrics,'selected_model') ?? pick(evaluation,'selected_model') ?? 'Logistic Regression';
 const modelScores = metrics?.selected_model_test_results ?? (evaluation?.results && evaluation.results[chosenName]) ?? metrics?.test_metrics ?? metrics?.held_out_test_performance ?? metrics?.test ?? evaluation ?? {};
 const cm=modelScores?.confusion_matrix || {};
 const validationRoot=validation?.evaluation ?? validation ?? {};
 const selectedTest=validationRoot?.selected_threshold_test_metrics || {};
 const thresholdRows=validationRoot?.threshold_analysis || validationRoot?.threshold_results || [];
 const barData=Object.entries(modelScores).filter(([k,v])=>['precision','recall','f1','f1_score','accuracy','roc_auc','average_precision','pr_auc'].includes(k)&&typeof v==='number').map(([name,value])=>({name:pretty(name),value:Number(value)}));
 const thresholdData=Array.isArray(thresholdRows)?thresholdRows:[];
 const shapChartData: {name:string;value:number}[] = Array.isArray(shap?.features)
   ? shap.features.filter((item:Json)=>typeof item.feature==='string' && typeof item.mean_abs_shap==='number' && Number.isFinite(item.mean_abs_shap))
       .map((item:Json)=>({name:pretty(item.feature),value:item.mean_abs_shap}))
   : [];
 const change=(k:keyof typeof initial, val:string)=>setForm(prev=>({...prev,[k]:typeof initial[k]==='number'?Number(val):val}));
 async function predict(e:React.FormEvent) { e.preventDefault();setPredicting(true);setPredictError('');setPrediction(null);
  try {const r=await fetch(`${API}/api/model/predict`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(form)});const body=await r.json();if(!r.ok)throw new Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail));setPrediction(body);}catch(e){setPredictError(e instanceof Error?e.message:String(e));}finally{setPredicting(false)}
 }
 return <div className="shell"><aside className={`sidebar ${menu?'show':''}`}><div className="brand"><HeartPulse size={27}/><div><b>Pregnancy Intelligence</b><span>Research Dashboard</span></div></div><div className="navlabel">WORKSPACE</div><nav>{pages.map(p=><button key={p.name} className={page===p.name?'active':''} onClick={()=>{setPage(p.name);setMenu(false)}}><p.icon size={18}/>{p.name}</button>)}</nav><div className="sidebar-bottom"><ShieldAlert size={17}/> Synthetic data · Research only</div></aside>
 <main className="main"><header className="topbar"><button className="mobile-menu" onClick={()=>setMenu(!menu)} aria-label="Toggle menu"><Menu/></button><span>Maternal & Newborn Health / {page}</span><span className="status"><i className={health?.status==='ok'?'online':''}/>{health?.status==='ok'?'API Connected':'API Not Confirmed'}</span></header>
 <div className="content"><div className="heading"><div><p className="eyebrow">PROJECT 02 · SYNTHETIC RESEARCH</p><h1>{page}</h1><p>High-Risk Pregnancy Outcomes — experimental model evaluation and interpretation</p></div></div>
 <div className="notice"><ShieldAlert size={19}/><span>{WARNING}</span></div>
 {loading&&<p>Loading backend data…</p>}{error&&<div className="error">{error}</div>}
 {page==='Overview'&&<><div className="metrics"><Metric label="Selected model" value={String(chosenName)} detail="Synthetic demonstration"/><Metric label="ROC-AUC" value={number(pick(modelScores,'roc_auc'))}/><Metric label="Average precision / PR-AUC" value={number(pick(modelScores,'average_precision','pr_auc'))}/><Metric label="Recall" value={percent(pick(modelScores,'recall'))}/></div><div className="two"><Panel title="Project Overview"><p>This independent portfolio project demonstrates the machine-learning workflow for a simulated hypertensive pregnancy complication outcome.</p><div className="detail"><b>Data source</b><span>Synthetic records only</span><b>Model</b><span>{String(chosenName)}</span><b>Clinical validation</b><span>Not performed</span><b>Prediction endpoint</b><span>Experimental only</span></div></Panel><Panel title="Model interpretation"><p>Performance metrics reflect a research evaluation on simulated data. A relatively high recall can coexist with many false positives and low precision.</p><p>Use the tabs to inspect performance, validation, explanations and the demonstration simulator.</p></Panel></div></>}
 {page==='Model Performance'&&<><div className="metrics"><Metric label="ROC-AUC" value={number(pick(modelScores,'roc_auc'))}/><Metric label="PR-AUC" value={number(pick(modelScores,'average_precision','pr_auc'))}/><Metric label="Precision" value={percent(pick(modelScores,'precision'))}/><Metric label="Recall" value={percent(pick(modelScores,'recall'))}/></div><div className="two"><Panel title="Performance measures">{barData.length?<div className="chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={barData} layout="vertical" margin={{left:20,right:20}}><CartesianGrid strokeDasharray="3 3" horizontal={false}/><XAxis type="number" domain={[0,1]}/><YAxis type="category" dataKey="name" width={125} tick={{fontSize:11}}/><Tooltip formatter={(v:any)=>number(Number(v))}/><Bar dataKey="value" fill="#2563eb" radius={[0,5,5,0]}/></BarChart></ResponsiveContainer></div>:<p>Metrics format not recognized. Review `/api/model/metrics`.</p>}</Panel><Panel title="Confusion Matrix"><div className="confusion"><div className="cm"><small>True negative</small><b>{cm.true_negative??'—'}</b></div><div className="cm"><small>False positive</small><b>{cm.false_positive??'—'}</b></div><div className="cm"><small>False negative</small><b>{cm.false_negative??'—'}</b></div><div className="cm"><small>True positive</small><b>{cm.true_positive??'—'}</b></div></div><p className="muted">Counts describe simulated records, not clinical outcomes.</p></Panel></div></>}
  {page==='Explainable AI'&&<><div className="two"><Panel title="SHAP global feature importance">{shapChartData.length?<><p className="muted">Mean absolute SHAP value for each feature. Higher values indicate a larger average contribution to the model's output on the explained synthetic samples.</p><div className="chart" style={{minHeight:Math.max(290,shapChartData.length*35)}}><ResponsiveContainer width="100%" height="100%"><BarChart data={shapChartData} layout="vertical" margin={{left:20,right:25,top:10,bottom:10}}><CartesianGrid strokeDasharray="3 3" horizontal={false}/><XAxis type="number"/><YAxis type="category" dataKey="name" width={160} tick={{fontSize:11}}/><Tooltip formatter={(v:any)=>number(Number(v),5)}/><Bar dataKey="value" name="Mean |SHAP|" fill="#2563eb" radius={[0,5,5,0]}/></BarChart></ResponsiveContainer></div></>:<p>{shapError||'No SHAP feature importance rows were returned. Check reports/shap_feature_importance.csv.'}</p>}</Panel><Panel title="Interpreting SHAP results"><p>SHAP describes how input features contribute to this trained model's output. It does not establish that any feature causes pregnancy complications.</p><p><b>Magnitude, not direction:</b> Mean absolute SHAP values rank average influence; they do not tell whether a feature raises or lowers a model score.</p><p><b>Scope:</b> The model was developed using synthetic records. Explanations are not clinically validated.</p>{shap?.source&&<p className="muted">Source: <code>{String(shap.source)}</code></p>}{shap?.value_scale&&<p className="muted">SHAP output scale: {String(shap.value_scale)}</p>}</Panel></div></>}
 {page==='Model Validation'&&<><div className="metrics"><Metric label="Selected threshold" value={number(pick(validationRoot,'selected_threshold'),2)}/><Metric label="Brier score" value={number(pick(validationRoot,'test_brier_score'))}/><Metric label="Test prevalence" value={percent(pick(validationRoot,'test_prevalence'))}/><Metric label="Selected-threshold recall" value={percent(pick(selectedTest,'recall'))}/></div><div className="two"><Panel title="Threshold analysis">{thresholdData.length?<div className="chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={thresholdData}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="threshold"/><YAxis domain={[0,1]}/><Tooltip/><Legend/><Line dataKey="precision" stroke="#2563eb" dot={false}/><Line dataKey="recall" stroke="#16a34a" dot={false}/><Line dataKey="f1" stroke="#f59e0b" dot={false}/></LineChart></ResponsiveContainer></div>:<p>{validation?'Validation summary loaded. Threshold-series data is not exposed by the current backend.':'Validation results unavailable. Finish Step 7 to create validation_metrics.json.'}</p>}</Panel><Panel title="Calibration and limitations"><p>Calibration compares model scores with observed outcome frequencies in synthetic held-out data.</p><p>A calibration curve is diagnostic; it does not recalibrate predictions. A threshold chosen by maximizing F1 is not a clinically approved decision threshold.</p></Panel></div></>}
 {page==='Prediction Simulator'&&<div className="two"><Panel title="Experimental input form"><p className="muted">Enter synthetic demonstration values. Categorical values must match your training dataset exactly.</p><form onSubmit={predict}><div className="formgrid">{numericFields.map(k=><label key={k}>{pretty(k)}<input type="number" required step="any" value={form[k]} onChange={e=>change(k,e.target.value)}/></label>)}{(['region','maternal_age_group','bmi_group'] as const).map(k=><label key={k}>{pretty(k)}<input required value={form[k]} placeholder="Training category value" onChange={e=>change(k,e.target.value)}/></label>)}</div><button className="primary" type="submit" disabled={predicting}>{predicting?'Calculating…':'Generate experimental score'}</button></form></Panel><Panel title="Model output">{predictError&&<div className="error">{predictError}</div>}{prediction?<><div className="result"><small>Experimental model score</small><strong>{typeof prediction.score_percentage==='number'?`${prediction.score_percentage.toFixed(2)}%`:number(prediction.experimental_model_score)}</strong></div><p>Threshold exceeded: <b>{prediction.threshold_exceeded?'Yes':'No'}</b></p><p className="muted">{prediction.disclaimer||WARNING}</p></>:<p>Submit a synthetic example to display the model output. The result is not a medical risk assessment.</p>}</Panel></div>}
 {page==='Research Limitations'&&<div className="two"><Panel title="Known limitations"><ul><li>Training and evaluation use simulated records.</li><li>No real-world clinical validation or external validation.</li><li>Probability calibration may be inadequate.</li><li>Model performance depends on synthetic outcome-generating assumptions.</li><li>Feature explanations are associations within the model, not causal findings.</li></ul></Panel><Panel title="Appropriate use"><p>Suitable for demonstrating data preprocessing, model comparison, explainability, threshold evaluation, API integration, and frontend development.</p><p><b>Not suitable for:</b> patient-specific assessment, screening, diagnosis, triage, or treatment decisions.</p></Panel></div>}
 <footer>Independent portfolio demonstration · Synthetic data only · No clinical use</footer></div></main></div>
}
