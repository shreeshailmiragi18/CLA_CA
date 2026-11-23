import React, { useState } from "react";
import axios from "axios";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter,
  CartesianGrid,
} from "recharts";
import {
  Upload,
  Activity,
  ShieldAlert,
  CheckCircle,
  Database,
  Layers,
  ArrowRight,
} from "lucide-react";

const COLORS = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"];

export default function App() {
  const [file, setFile] = useState(null);
  const [bim, setBim] = useState(40);
  const [alpha, setAlpha] = useState(0.8);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const readFileAsBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  const handleRun = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a CSV file first.");
      return;
    }
    setError(null);
    setLoading(true);

    try {
      const base64 = await readFileAsBase64(file);
      // Strip the "data:application/csv;base64," prefix if present
      const b64Data = base64.split(",")[1] || base64;

      const response = await axios.post(
        `http://localhost:8000/api/run-experiment?bim=${bim}&alpha=${alpha}`,
        { csv_base64: b64Data },
        { headers: { "Content-Type": "application/json" } },
      );
      setResult(response.data);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          err.message ||
          "An error occurred during experiment execution.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col font-sans">
      {/* Header */}
      <header className="bg-slate-800 border-b border-slate-700 px-6 py-4 flex justify-between items-center shadow-md">
        <div className="flex items-center space-x-3">
          <Activity className="w-8 h-8 text-blue-500 animate-pulse" />
          <h1 className="text-xl font-bold tracking-wide">
            Software Defect Predection Using CLA-CA and Ensemble Learning
          </h1>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 p-6 max-w-7xl mx-auto w-full space-y-6">
        {/* Upload & Config Section */}
        <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-lg">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Upload className="w-5 h-5 text-blue-400" /> Experiment
            Configuration & Dataset Upload
          </h2>
          <form
            onSubmit={handleRun}
            className="grid grid-cols-1 md:grid-cols-4 gap-6 items-end"
          >
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Select Dataset (CSV)
              </label>
              <input
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className="block w-full text-sm text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700 cursor-pointer"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Bim (Discretization Regions): {bim}
              </label>
              <input
                type="number"
                value={bim}
                onChange={(e) => setBim(Number(e.target.value))}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Alpha (Amplified Coeff): {alpha}
              </label>
              <input
                type="number"
                step="0.1"
                value={alpha}
                onChange={(e) => setAlpha(Number(e.target.value))}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-100 focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <button
                type="submit"
                disabled={loading}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2.5 px-4 rounded-lg transition duration-200 flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {loading ? "Running..." : "Run Pipeline"}{" "}
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </form>
          {error && (
            <div className="mt-4 p-3 bg-red-900/50 border border-red-700 rounded-lg text-red-200 text-sm">
              {error}
            </div>
          )}
        </div>

        {/* Results Dashboard */}
        {result && (
          <div className="space-y-6">
            {/* KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="bg-slate-800 border border-slate-700 rounded-xl p-5 shadow-lg flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-400">
                    Total Modules
                  </p>
                  <h3 className="text-2xl font-bold mt-1 text-white">
                    {result.total_modules}
                  </h3>
                </div>
                <Database className="w-10 h-10 text-blue-400 opacity-80" />
              </div>
              <div className="bg-slate-800 border border-slate-700 rounded-xl p-5 shadow-lg flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-400">
                    Defective Modules
                  </p>
                  <h3 className="text-2xl font-bold mt-1 text-red-400">
                    {result.defective_count}
                  </h3>
                </div>
                <ShieldAlert className="w-10 h-10 text-red-400 opacity-80" />
              </div>
              <div className="bg-slate-800 border border-slate-700 rounded-xl p-5 shadow-lg flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-400">
                    Non-Defective Modules
                  </p>
                  <h3 className="text-2xl font-bold mt-1 text-emerald-400">
                    {result.non_defective_count}
                  </h3>
                </div>
                <CheckCircle className="w-10 h-10 text-emerald-400 opacity-80" />
              </div>
              <div className="bg-slate-800 border border-slate-700 rounded-xl p-5 shadow-lg flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-400">
                    Retained Features
                  </p>
                  <h3 className="text-2xl font-bold mt-1 text-purple-400">
                    {result.retained_features_count}
                  </h3>
                </div>
                <Layers className="w-10 h-10 text-purple-400 opacity-80" />
              </div>
            </div>

            {/* Metrics & Charts Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Evaluation Metrics Card */}
              <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-lg">
                <h3 className="text-lg font-semibold mb-4 text-slate-200">
                  Evaluation Metrics
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-slate-900 p-4 rounded-lg border border-slate-700/50">
                    <p className="text-sm text-slate-400">F1-Score</p>
                    <p className="text-xl font-bold text-blue-400 mt-1">
                      {result.metrics.f1.toFixed(4)}
                    </p>
                  </div>
                  <div className="bg-slate-900 p-4 rounded-lg border border-slate-700/50">
                    <p className="text-sm text-slate-400">Recall</p>
                    <p className="text-xl font-bold text-emerald-400 mt-1">
                      {result.metrics.recall.toFixed(4)}
                    </p>
                  </div>
                  <div className="bg-slate-900 p-4 rounded-lg border border-slate-700/50">
                    <p className="text-sm text-slate-400">Accuracy</p>
                    <p className="text-xl font-bold text-purple-400 mt-1">
                      {result.metrics.accuracy.toFixed(4)}
                    </p>
                  </div>
                  <div className="bg-slate-900 p-4 rounded-lg border border-slate-700/50">
                    <p className="text-sm text-slate-400">MCC</p>
                    <p className="text-xl font-bold text-amber-400 mt-1">
                      {result.metrics.mcc.toFixed(4)}
                    </p>
                  </div>
                </div>
              </div>

              {/* Defect Distribution Pie Chart */}
              <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-lg flex flex-col items-center justify-center">
                <h3 className="text-lg font-semibold mb-2 text-slate-200 self-start">
                  Defect Distribution
                </h3>
                <div className="w-full h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={[
                          { name: "Defective", value: result.defective_count },
                          {
                            name: "Non-Defective",
                            value: result.non_defective_count,
                          },
                        ]}
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                        label
                      >
                        <Cell fill="#ef4444" />
                        <Cell fill="#10b981" />
                      </Pie>
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#1e293b",
                          borderColor: "#475569",
                          borderRadius: "8px",
                        }}
                      />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* Feature Weights Chart */}
            <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 text-slate-200">
                Top Feature Weights (Algorithm 2)
              </h3>
              <div className="w-full h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={Object.entries(result.weights).map(([k, v]) => ({
                      feature: k,
                      weight: v,
                    }))}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis
                      dataKey="feature"
                      stroke="#94a3b8"
                      angle={-30}
                      textAnchor="end"
                      height={60}
                      interval={0}
                      tick={{ fontSize: 10 }}
                    />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#1e293b",
                        borderColor: "#475569",
                        borderRadius: "8px",
                      }}
                    />
                    <Bar
                      dataKey="weight"
                      fill="#3b82f6"
                      radius={[4, 4, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Modules Table Preview */}
            <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 shadow-lg">
              <h3 className="text-lg font-semibold mb-4 text-slate-200">
                Modules Prediction Preview
              </h3>
              <div className="overflow-x-auto max-h-96">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-slate-700 text-slate-400 text-sm">
                      <th className="py-3 px-4">Module ID</th>
                      <th className="py-3 px-4">Score ($k_i$)</th>
                      <th className="py-3 px-4">Predicted Label</th>
                      <th className="py-3 px-4">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-700/50 text-sm">
                    {result.modules.map((m) => (
                      <tr key={m.id} className="hover:bg-slate-700/30">
                        <td className="py-3 px-4 font-mono">#{m.id}</td>
                        <td className="py-3 px-4 font-mono">
                          {m.score.toFixed(3)}
                        </td>
                        {/* <td className="py-3 px-4">
                          {m.actual === 1 ? "Defective" : "Clean"}
                        </td> */}
                        <td className="py-3 px-4">
                          {m.predicted === 1 ? "Defective" : "Clean"}
                        </td>
                        <td className="py-3 px-4">
                          <span
                            className={`px-2.5 py-1 rounded-full text-xs font-semibold ${
                              m.status === "Defective"
                                ? "bg-red-900/50 text-red-300 border border-red-700"
                                : "bg-emerald-900/50 text-emerald-300 border border-emerald-700"
                            }`}
                          >
                            {m.status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
