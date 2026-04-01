import { ComputerDesktopIcon } from './Icons'

export default function MachineList({ machines, onSelectMachine, selectedMachine }) {
  const getUsageColor = (usage) => {
    if (usage >= 90) return 'text-red-400 font-bold'
    if (usage >= 80) return 'text-yellow-400 font-semibold'
    return 'text-green-400'
  }

  const calculateHealthScore = (machine) => {
    const cpu = parseFloat(machine.cpu_usage)
    const memory = parseFloat(machine.memory_usage)
    const disk = parseFloat(machine.disk_usage)
    
    // Health score: 100 - average usage
    const avgUsage = (cpu + memory + disk) / 3
    const score = Math.max(0, 100 - avgUsage)
    return score.toFixed(0)
  }

  const calculateRisk = (machine) => {
    // Use RF prediction risk_level if available
    if (machine.prediction && machine.prediction.risk_level) {
      const riskMap = {
        CRITICAL: { label: 'Critique', color: 'text-red-400 font-bold',    bg: 'bg-red-500/20',    showIcon: true  },
        HIGH:     { label: 'Élevé',    color: 'text-orange-400 font-bold', bg: 'bg-orange-500/20', showIcon: false },
        MEDIUM:   { label: 'Moyen',    color: 'text-yellow-400 font-semibold', bg: 'bg-yellow-500/20', showIcon: false },
        LOW:      { label: 'Faible',   color: 'text-green-400',            bg: 'bg-green-500/20',  showIcon: false },
      }
      return riskMap[machine.prediction.risk_level] || { label: 'N/A', color: 'text-gray-400', bg: 'bg-gray-500/20', showIcon: false }
    }
    
    // Fallback to heuristic
    const cpu = parseFloat(machine.cpu_usage)
    const memory = parseFloat(machine.memory_usage)
    const disk = parseFloat(machine.disk_usage)
    const maxUsage = Math.max(cpu, memory, disk)

    if (maxUsage >= 90 || machine.health_status === 'CRITICAL') {
      return { label: 'Élevé', color: 'text-red-400 font-bold', bg: 'bg-red-500/20', showIcon: true }
    }
    if (maxUsage >= 80 || machine.health_status === 'WARNING') {
      return { label: 'Moyen', color: 'text-yellow-400 font-semibold', bg: 'bg-yellow-500/20', showIcon: false }
    }
    return { label: 'Faible', color: 'text-green-400', bg: 'bg-green-500/20', showIcon: false }
  }

  const getProbaColor = (proba) => {
    if (proba >= 70) return 'text-red-400 font-bold'
    if (proba >= 50) return 'text-orange-400 font-semibold'
    if (proba >= 30) return 'text-yellow-400'
    return 'text-green-400'
  }

  return (
    <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-2xl shadow-xl overflow-hidden">
      <div className="px-6 py-4 border-b border-white/10 bg-gradient-to-r from-white/5 to-transparent">
        <h2 className="text-lg font-bold text-white flex items-center space-x-2">
          <ComputerDesktopIcon className="w-6 h-6" />
          <span>Machines Critiques</span>
        </h2>
        <p className="text-sm text-gray-300 mt-1">{machines.length} machines surveillées</p>
      </div>
      
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-white/10">
          <thead className="bg-white/5">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">
                PC-ID
              </th>
              <th className="px-6 py-3 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">
                CPU %
              </th>
              <th className="px-6 py-3 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">
                RAM %
              </th>
              <th className="px-6 py-3 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">
                Disque %
              </th>
              <th className="px-6 py-3 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">
                Score Santé
              </th>
              <th className="px-6 py-3 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">
                Proba. Panne (RF)
              </th>
              <th className="px-6 py-3 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">
                Santé Disque (LSTM)
              </th>
              <th className="px-6 py-3 text-left text-xs font-bold text-gray-300 uppercase tracking-wider">
                Action
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/10">
            {machines.map((machine) => {
              const healthScore = calculateHealthScore(machine)
              
              return (
                <tr
                  key={machine.id}
                  className={`hover:bg-white/5 transition cursor-pointer ${
                    selectedMachine?.id === machine.id ? 'bg-white/10' : ''
                  }`}
                  onClick={() => onSelectMachine(machine)}
                >
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className={`w-2 h-2 rounded-full mr-2 ${
                        machine.status === 'online' ? 'bg-green-400' : 'bg-gray-500'
                      }`}></div>
                      <div>
                        <div className="text-sm font-bold text-white">{machine.hostname}</div>
                        <div className="text-xs text-gray-400">{machine.ip_address}</div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`text-sm font-semibold ${getUsageColor(parseFloat(machine.cpu_usage))}`}>
                      {parseFloat(machine.cpu_usage).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`text-sm font-semibold ${getUsageColor(parseFloat(machine.memory_usage))}`}>
                      {parseFloat(machine.memory_usage).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`text-sm font-semibold ${getUsageColor(parseFloat(machine.disk_usage))}`}>
                      {parseFloat(machine.disk_usage).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="w-16 bg-white/10 rounded-full h-2 mr-2">
                        <div 
                          className={`h-2 rounded-full ${
                            healthScore >= 70 ? 'bg-green-500' : 
                            healthScore >= 50 ? 'bg-yellow-500' : 'bg-red-500'
                          }`}
                          style={{ width: `${healthScore}%` }}
                        ></div>
                      </div>
                      <span className="text-sm font-semibold text-white">{healthScore}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {machine.prediction && machine.prediction.failure_probability_30d != null && machine.prediction.failure_probability_30d > 0 ? (
                      <span className={`text-sm ${getProbaColor(machine.prediction.failure_probability_30d)}`}>
                        {parseFloat(machine.prediction.failure_probability_30d).toFixed(1)}%
                        <span className="text-xs text-gray-400 ml-1">/ 30j</span>
                      </span>
                    ) : (
                      <span className="text-sm text-gray-500">N/A</span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {machine.lstmPrediction && machine.lstmPrediction.prediction !== null ? (
                      <span className={`text-sm font-semibold ${
                        machine.lstmPrediction.prediction >= 0.7 ? 'text-red-400' :
                        machine.lstmPrediction.prediction >= 0.5 ? 'text-orange-400' :
                        machine.lstmPrediction.prediction >= 0.3 ? 'text-yellow-400' :
                        'text-green-400'
                      }`}>
                        {Math.round(machine.lstmPrediction.prediction * 100)}%
                      </span>
                    ) : machine.lstmPrediction ? (
                      <span className="text-xs text-gray-400">Données insuffisantes</span>
                    ) : (
                      <span className="text-sm text-gray-500">N/A</span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        onSelectMachine(machine)
                      }}
                      className="px-3 py-1 bg-gradient-to-r from-purple-500 to-indigo-500 hover:from-purple-600 hover:to-indigo-600 text-white text-sm rounded-lg transition shadow-lg shadow-purple-500/30"
                    >
                      Détails
                    </button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
