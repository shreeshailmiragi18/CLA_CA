import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';

const data = [
  {"Accuracy":0.587,"Dataset":"ant-1.5","F1":0.301,"MCC":0.232,"Precision":0.184,"Recall":0.813},
  {"Accuracy":0.573,"Dataset":"arc","F1":0.294,"MCC":0.165,"Precision":0.187,"Recall":0.690},
  {"Accuracy":0.837,"Dataset":"berek","F1":0.811,"MCC":0.692,"Precision":0.714,"Recall":0.938},
  {"Accuracy":0.655,"Dataset":"camel-1.0","F1":0.146,"MCC":0.167,"Precision":0.081,"Recall":0.769},
  {"Accuracy":0.622,"Dataset":"ivy-2.0","F1":0.296,"MCC":0.201,"Precision":0.188,"Recall":0.700},
  {"Accuracy":0.743,"Dataset":"jedit-3.2","F1":0.646,"MCC":0.451,"Precision":0.593,"Recall":0.711},
  {"Accuracy":0.577,"Dataset":"synapse-1.1","F1":0.420,"MCC":0.131,"Precision":0.333,"Recall":0.567},
  {"Accuracy":0.628,"Dataset":"tomcat","F1":0.283,"MCC":0.247,"Precision":0.171,"Recall":0.818},
  {"Accuracy":0.631,"Dataset":"xalan-2.4","F1":0.375,"MCC":0.247,"Precision":0.252,"Recall":0.727},
  {"Accuracy":0.573,"Dataset":"xerces-1.2","F1":0.203,"MCC":-0.033,"Precision":0.145,"Recall":0.338},
  {"Accuracy":0.744,"Dataset":"EQ","F1":0.713,"MCC":0.496,"Precision":0.644,"Recall":0.798},
  {"Accuracy":0.645,"Dataset":"JDT","F1":0.488,"MCC":0.340,"Precision":0.348,"Recall":0.820},
  {"Accuracy":0.601,"Dataset":"LC","F1":0.277,"MCC":0.236,"Precision":0.167,"Recall":0.828},
  {"Accuracy":0.551,"Dataset":"ML","F1":0.264,"MCC":0.104,"Precision":0.168,"Recall":0.612},
  {"Accuracy":0.578,"Dataset":"PDE","F1":0.322,"MCC":0.190,"Precision":0.208,"Recall":0.718}
];

const PerformanceVisualization = () => (
  <div style={{ width: '100%', height: 500, padding: 20 }}>
    <h3>Software Defect Prediction Performance (CLA-CA)</h3>
    <ResponsiveContainer>
      <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="Dataset" angle={-45} textAnchor="end" height={80} />
        <YAxis domain={[0, 1]} />
        <Tooltip />
        <Legend />
        <Bar dataKey="Recall" fill="#8884d8" />
        <Bar dataKey="F1" fill="#82ca9d" />
        <Bar dataKey="Accuracy" fill="#ffc658" />
      </BarChart>
    </ResponsiveContainer>
  </div>
);

export default PerformanceVisualization;
