import { useState, useEffect } from 'react'
import axios from 'axios'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { CpuChipIcon, ArrowPathIcon } from './Icons'

export default function ModelPerformance({ onClose }) {
  const [lstmModels, setLstmModels] = useState([])
  const [rfModels, setRfModels] = useState([])
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('lstm')

  useEffect(() => { fetchModels() }, [])

  const fetchModels = async () => {
    setLoading(true)
    try {
      const token = localStorage.getItem('token')
      const res = await axios.get('http://localhost:3000/api/ml/models', {
        headers: { Authorization: `Bearer ${token}` }
      })
      const all = res.data.models || []
      setLstmModels(all.filter(m => m.model_type === 'lstm'))
      setRfModels(all.filter(m => m.model_type === 'random_forest'))
    } catch (e) {
      console.error('Error fetching models:', e)
    } finally {
      setLoading(false)
    }
  }

  const fmt = (v) => v != null ? parseFloat(v).toFixed(4) : '—'
  const fmtPct = (v) => v != null ? (parseFloat(v) * 100).toFixed(1) + '%' : '—'

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl px-4 py-2 text-white text-sm shadow-xl">
          <p className="font-semibold">{label}</p>
          <p className="text-purple-300">{payload[0].value.toFixed(4)}</p>
        </div>
      )
    }
    return null
  }

  // ── LSTM tab ──────────────────────────────────────────────────────────────
  const LSTMTab = () => {
    const activeModel = lstmModels.find(m => m.is_active) || lstmModels[0]
    const metricsData = activeModel ? [
      { name: 'MAE',  value: parseFloat(activeModel.mae  || 0) },
      { name: 'MSE',  value: parseFloat(activeModel.mse  || 0) },
      { name: 'RMSE', value: parseFloat(activeModel.rmse || 0) },
    ] : []

    if (!activeModel) return (
      <div className="h-48 flex flex-col items-center justify-center text-gray-300 space-y-3">
        <p>Aucun modèle LSTM trouvé.</p>
        <code className="bg-white/10 border border-white/20 px-2 py-1 rounded-lg text-xs text-purple-300">
          node backend/seed-lstm-model.js
        </code>
      </div>
    )

    return (
      <>
        <div className="mb-5 bg-white/10 backdrop-blur-md border border-white/20 rounded-2xl p-6">
          <div className="flex justify-between items-start mb-5">
            <div>
              <h3 className="text-lg font-bold text-white">Modèle Actif</h3>
              <p className="text-sm text-gray-300 mt-1">ID : <span className="font-mono font-semibold text-purple-300">{activeModel.model_id}</span></p>
              <p className="text-sm text-gray-300">Entraîné le : {new Date(activeModel.trained_at).toLocaleString('fr-FR')}</p>
            </div>
            <button onClick={fetchModels} className="px-4 py-2 bg-gradient-to-r from-purple-500 to-indigo-500 hover:from-purple-600 hover:to-indigo-600 text-white rounded-xl flex items-center space-x-2 text-sm transition-all shadow-lg shadow-purple-500/30">
              <ArrowPathIcon className="w-4 h-4" /><span>Actualiser</span>
            </button>
          </div>
          <div className="grid grid-cols-3 gap-4">
            {[
              { label: 'MAE',  value: fmt(activeModel.mae),  color: 'text-blue-400',   sub: 'Mean Absolute Error' },
              { label: 'MSE',  value: fmt(activeModel.mse),  color: 'text-purple-400', sub: 'Mean Squared Error' },
              { label: 'RMSE', value: fmt(activeModel.rmse), color: 'text-green-400',  sub: 'Root Mean Squared Error' },
            ].map(({ label, value, color, sub }) => (
              <div key={label} className="bg-white/10 backdrop-blur-sm border border-white/20 rounded-xl p-4 text-center">
                <p className="text-xs text-gray-300 mb-1 font-medium uppercase tracking-wide">{label}</p>
                <p className={`text-2xl font-bold ${color}`}>{value}</p>
                <p className="text-xs text-gray-400 mt-1">{sub}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="mb-5 bg-white/10 backdrop-blur-md border border-white/20 rounded-2xl p-6">
          <h3 className="text-base font-bold text-white mb-4">Visualisation des Métriques</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={metricsData} barSize={48}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="name" tick={{ fill: '#d1d5db', fontSize: 13 }} axisLine={{ stroke: 'rgba(255,255,255,0.2)' }} tickLine={false} />
              <YAxis tickFormatter={v => v.toFixed(3)} tick={{ fill: '#d1d5db', fontSize: 11 }} axisLine={{ stroke: 'rgba(255,255,255,0.2)' }} tickLine={false} />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.05)' }} />
              <Bar dataKey="value" fill="url(#barGradient)" radius={[6, 6, 0, 0]} />
              <defs>
                <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#a78bfa" />
                  <stop offset="100%" stopColor="#6366f1" />
                </linearGradient>
              </defs>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {lstmModels.length > 0 && (
          <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-2xl overflow-hidden">
            <div className="px-6 py-3 border-b border-white/20">
              <h3 className="text-sm font-bold text-white">Historique des modèles LSTM</h3>
            </div>
            <table className="min-w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  {['Version', 'MAE', 'MSE', 'RMSE', 'Date', 'Statut'].map(h => (
                    <th key={h} className="px-4 py-3 text-left text-xs font-semibold text-gray-300 uppercase tracking-wider">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {lstmModels.map((m, i) => (
                  <tr key={i} className={`border-b border-white/10 transition-colors ${m.is_active ? 'bg-purple-500/20' : 'hover:bg-white/5'}`}>
                    <td className="px-4 py-3 font-mono font-semibold text-purple-300">v{m.version}</td>
                    <td className="px-4 py-3 text-blue-400 font-medium">{fmt(m.mae)}</td>
                    <td className="px-4 py-3 text-purple-400 font-medium">{fmt(m.mse)}</td>
                    <td className="px-4 py-3 text-green-400 font-medium">{fmt(m.rmse)}</td>
                    <td className="px-4 py-3 text-gray-300">{new Date(m.trained_at).toLocaleDateString('fr-FR')}</td>
                    <td className="px-4 py-3">
                      {m.is_active
                        ? <span className="px-2.5 py-1 text-xs font-semibold rounded-full bg-green-500/20 text-green-400 border border-green-500/30">Actif</span>
                        : <span className="px-2.5 py-1 text-xs font-semibold rounded-full bg-white/10 text-gray-400 border border-white/20">Inactif</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </>
    )
  }

  // ── RF tab ────────────────────────────────────────────────────────────────
  const RFTab = () => {
    const activeModel = rfModels.find(m => m.is_active) || rfModels[0]
    const metricsData = activeModel ? [
      { name: 'Accuracy',  value: parseFloat(activeModel.accuracy  || 0) },
      { name: 'Précision', value: parseFloat(activeModel.precision_score || 0) },
      { name: 'Rappel',    value: parseFloat(activeModel.recall    || 0) },
      { name: 'F1-Score',  value: parseFloat(activeModel.f1_score  || 0) },
    ] : []

    if (!activeModel) return (
      <div className="h-48 flex flex-col items-center justify-center text-gray-300 space-y-3">
        <p>Aucun modèle Random Forest trouvé.</p>
        <p className="text-sm text-gray-400">Lancez un entraînement depuis le service ML.</p>
      </div>
    )

    return (
      <>
        <div className="mb-5 bg-white/10 backdrop-blur-md border border-white/20 rounded-2xl p-6">
          <div className="flex justify-between items-start mb-5">
            <div>
              <h3 className="text-lg font-bold text-white">Modèle Actif</h3>
              <p className="text-sm text-gray-300 mt-1">ID : <span className="font-mono font-semibold text-green-300">{activeModel.model_id}</span></p>
              <p className="text-sm text-gray-300">Entraîné le : {new Date(activeModel.trained_at).toLocaleString('fr-FR')}</p>
            </div>
            <button onClick={fetchModels} className="px-4 py-2 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white rounded-xl flex items-center space-x-2 text-sm transition-all shadow-lg shadow-green-500/30">
              <ArrowPathIcon className="w-4 h-4" /><span>Actualiser</span>
            </button>
          </div>
          <div className="grid grid-cols-4 gap-4">
            {[
              { label: 'Accuracy',  value: fmtPct(activeModel.accuracy),          color: 'text-blue-400' },
              { label: 'Précision', value: fmtPct(activeModel.precision_score),   color: 'text-green-400' },
              { label: 'Rappel',    value: fmtPct(activeModel.recall),            color: 'text-yellow-400' },
              { label: 'F1-Score',  value: fmtPct(activeModel.f1_score),          color: 'text-purple-400' },
            ].map(({ label, value, color }) => (
              <div key={label} className="bg-white/10 backdrop-blur-sm border border-white/20 rounded-xl p-4 text-center">
                <p className="text-xs text-gray-300 mb-1 font-medium uppercase tracking-wide">{label}</p>
                <p className={`text-2xl font-bold ${color}`}>{value}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="mb-5 bg-white/10 backdrop-blur-md border border-white/20 rounded-2xl p-6">
          <h3 className="text-base font-bold text-white mb-4">Visualisation des Métriques</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={metricsData} barSize={48}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="name" tick={{ fill: '#d1d5db', fontSize: 12 }} axisLine={{ stroke: 'rgba(255,255,255,0.2)' }} tickLine={false} />
              <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fill: '#d1d5db', fontSize: 11 }} axisLine={{ stroke: 'rgba(255,255,255,0.2)' }} tickLine={false} />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.05)' }} />
              <Bar dataKey="value" fill="url(#rfGradient)" radius={[6, 6, 0, 0]} />
              <defs>
                <linearGradient id="rfGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#34d399" />
                  <stop offset="100%" stopColor="#059669" />
                </linearGradient>
              </defs>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {rfModels.length > 0 && (
          <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-2xl overflow-hidden">
            <div className="px-6 py-3 border-b border-white/20">
              <h3 className="text-sm font-bold text-white">Historique des modèles Random Forest</h3>
            </div>
            <table className="min-w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  {['Version', 'Accuracy', 'Précision', 'Rappel', 'F1', 'Date', 'Statut'].map(h => (
                    <th key={h} className="px-4 py-3 text-left text-xs font-semibold text-gray-300 uppercase tracking-wider">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rfModels.map((m, i) => (
                  <tr key={i} className={`border-b border-white/10 transition-colors ${m.is_active ? 'bg-green-500/20' : 'hover:bg-white/5'}`}>
                    <td className="px-4 py-3 font-mono font-semibold text-green-300">v{m.version}</td>
                    <td className="px-4 py-3 text-blue-400 font-medium">{fmtPct(m.accuracy)}</td>
                    <td className="px-4 py-3 text-green-400 font-medium">{fmtPct(m.precision_score)}</td>
                    <td className="px-4 py-3 text-yellow-400 font-medium">{fmtPct(m.recall)}</td>
                    <td className="px-4 py-3 text-purple-400 font-medium">{fmtPct(m.f1_score)}</td>
                    <td className="px-4 py-3 text-gray-300">{new Date(m.trained_at).toLocaleDateString('fr-FR')}</td>
                    <td className="px-4 py-3">
                      {m.is_active
                        ? <span className="px-2.5 py-1 text-xs font-semibold rounded-full bg-green-500/20 text-green-400 border border-green-500/30">Actif</span>
                        : <span className="px-2.5 py-1 text-xs font-semibold rounded-full bg-white/10 text-gray-400 border border-white/20">Inactif</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </>
    )
  }

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-2xl shadow-2xl shadow-purple-900/50 max-w-4xl w-full max-h-[90vh] flex flex-col">

        {/* Header */}
        <div className="px-6 py-4 border-b border-white/20 flex justify-between items-center bg-white/20 backdrop-blur-md rounded-t-2xl flex-shrink-0">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/30">
              <CpuChipIcon className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">Performance des Modèles ML</h2>
              <p className="text-sm text-gray-300">Random Forest · LSTM Backblaze</p>
            </div>
          </div>
          <button onClick={onClose} className="text-white/60 hover:text-white hover:bg-white/10 rounded-xl w-9 h-9 flex items-center justify-center text-2xl transition-all">×</button>
        </div>

        {/* Tabs */}
        <div className="px-6 pt-4 flex gap-2 flex-shrink-0">
          {[
            { key: 'lstm', label: 'LSTM — Backblaze', color: 'from-purple-500 to-indigo-500' },
            { key: 'rf',   label: 'Random Forest',    color: 'from-green-500 to-emerald-500' },
          ].map(({ key, label, color }) => (
            <button
              key={key}
              onClick={() => setActiveTab(key)}
              className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${
                activeTab === key
                  ? `bg-gradient-to-r ${color} text-white shadow-lg`
                  : 'bg-white/10 text-gray-300 hover:bg-white/20 border border-white/10'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="overflow-y-auto max-h-[75vh] px-6 py-5 custom-scrollbar">
          {loading ? (
            <div className="h-48 flex items-center justify-center text-gray-300">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '0.15s' }}></div>
                <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0.3s' }}></div>
                <span className="ml-2">Chargement...</span>
              </div>
            </div>
          ) : activeTab === 'lstm' ? <LSTMTab /> : <RFTab />}
        </div>
      </div>
    </div>
  )
}
