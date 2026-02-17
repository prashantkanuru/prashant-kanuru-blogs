const ADAnimation = () => {
    const [step, setStep] = React.useState(0);
    const [mode, setMode] = React.useState('forward'); // 'forward' or 'reverse'
    const [inputs, setInputs] = React.useState({ x1: 2, x2: 3 });

    // Computational Graph: y = sin(x1 * x2)
    // Nodes:
    // v_ -1 = x1
    // v_0 = x2
    // v_1 = x1 * x2 (w)
    // v_2 = sin(v_1) (y)

    const values = {
        x1: inputs.x1,
        x2: inputs.x2,
        w: inputs.x1 * inputs.x2,
        y: Math.sin(inputs.x1 * inputs.x2)
    };

    // Gradients for VJP (dL/dx) - assuming dL/dy = 1
    const grads = {
        y: 1,
        w: Math.cos(values.w) * 1,
        x1: values.x2 * (Math.cos(values.w) * 1),
        x2: values.x1 * (Math.cos(values.w) * 1)
    };

    // Tangents for JVP (dy/dx) - calculating dy/dx1 (seed x1=1, x2=0)
    const tangents = {
        x1: 1,
        x2: 0,
        w: 1 * values.x2 + values.x1 * 0, // x2
        y: Math.cos(values.w) * values.x2
    };

    const stepsInfo = {
        forward: [
            { text: "Start: Inputs x1, x2", highlight: ['x1', 'x2'] },
            { text: "Step 1: w = x1 * x2", highlight: ['w'], arrow: 'mult' },
            { text: "Step 2: y = sin(w)", highlight: ['y'], arrow: 'sin' }
        ],
        reverse: [
            { text: "Start: Gradient at Output (dL/dy = 1)", highlight: ['y'] },
            { text: "Step 1: Pullback through sin. dL/dw = cos(w) * dL/dy", highlight: ['w'], arrow: 'sin_back' },
            { text: "Step 2: Pullback through mult. dL/dx1 = x2 * dL/dw", highlight: ['x1'], arrow: 'mult_back1' },
            { text: "Step 3: Pullback through mult. dL/dx2 = x1 * dL/dw", highlight: ['x2'], arrow: 'mult_back2' }
        ]
    };

    const handleNext = () => {
        setStep(Math.min(step + 1, stepsInfo[mode].length - 1));
    };

    const handlePrev = () => {
        setStep(Math.max(step - 1, 0));
    };

    const toggleMode = (newMode) => {
        setMode(newMode);
        setStep(0);
    };

    const Node = ({ label, value, id, x, y, isInput, grad, tangent }) => {
        const isActive = stepsInfo[mode][step].highlight.includes(id) ||
            (mode === 'forward' && step >= stepsInfo[mode].findIndex(s => s.highlight.includes(id))) || // Keep lit in forward
            (mode === 'reverse' && step >= stepsInfo[mode].findIndex(s => s.highlight.includes(id)));  // Keep lit in reverse

        return (
            <g transform={`translate(${x}, ${y})`}>
                <circle
                    cx="0" cy="0" r="30"
                    fill={isActive ? "#6366f1" : "#1e293b"}
                    stroke={isActive ? "#a855f7" : "#475569"}
                    strokeWidth="2"
                />
                <text x="0" y="-5" textAnchor="middle" fill="#fff" fontSize="14" fontWeight="bold">{label}</text>
                <text x="0" y="15" textAnchor="middle" fill="#cbd5e1" fontSize="12">
                    {value.toFixed(2)}
                </text>
                {mode === 'reverse' && step > 0 && isActive && (
                    <text x="0" y="45" textAnchor="middle" fill="#ef4444" fontSize="12" fontWeight="bold">
                        ∇:{grad ? grad.toFixed(2) : 0}
                    </text>
                )}
                {mode === 'forward' && isActive && (
                    <text x="0" y="45" textAnchor="middle" fill="#22c55e" fontSize="12" fontWeight="bold">
                        val: {value.toFixed(2)}
                    </text>
                )}
            </g>
        );
    };

    return (
        <div style={{ fontFamily: 'Inter, sans-serif', background: 'rgba(30, 41, 59, 0.5)', padding: '20px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.1)', color: '#f8fafc', marginTop: '20px' }}>
            <h3 style={{ marginTop: 0, color: '#a855f7' }}>AD Visualization: y = sin(x1 * x2)</h3>

            <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
                <button
                    onClick={() => toggleMode('forward')}
                    style={{
                        padding: '8px 16px',
                        background: mode === 'forward' ? '#6366f1' : '#334155',
                        border: 'none',
                        borderRadius: '6px',
                        color: 'white',
                        cursor: 'pointer'
                    }}
                >
                    Forward Mode (Primal)
                </button>
                <button
                    onClick={() => toggleMode('reverse')}
                    style={{
                        padding: '8px 16px',
                        background: mode === 'reverse' ? '#ef4444' : '#334155',
                        border: 'none',
                        borderRadius: '6px',
                        color: 'white',
                        cursor: 'pointer'
                    }}
                >
                    Reverse Mode (Gradient)
                </button>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
                <div>
                    Steps: {stepsInfo[mode][step].text}
                </div>
                <div>
                    <button onClick={handlePrev} disabled={step === 0} style={{ marginRight: '10px', padding: '5px 10px', borderRadius: '4px', border: 'none', cursor: 'pointer', background: '#475569', color: 'white', opacity: step === 0 ? 0.5 : 1 }}>Prev</button>
                    <button onClick={handleNext} disabled={step === stepsInfo[mode].length - 1} style={{ padding: '5px 10px', borderRadius: '4px', border: 'none', cursor: 'pointer', background: '#6366f1', color: 'white', opacity: step === stepsInfo[mode].length - 1 ? 0.5 : 1 }}>Next</button>
                </div>
            </div>

            <svg viewBox="0 0 400 300" style={{ width: '100%', height: '300px', background: '#0f172a', borderRadius: '8px' }}>
                {/* Edges */}
                <line x1="80" y1="100" x2="200" y2="150" stroke="#475569" strokeWidth="2" />
                <line x1="80" y1="200" x2="200" y2="150" stroke="#475569" strokeWidth="2" />
                <line x1="200" y1="150" x2="320" y2="150" stroke="#475569" strokeWidth="2" />

                {/* Arrowheads/Flow indicators could go here */}

                {/* Nodes */}
                <Node label="x1" value={values.x1} id="x1" x={80} y={100} grad={grads.x1} />
                <Node label="x2" value={values.x2} id="x2" x={80} y={200} grad={grads.x2} />
                <Node label="w (x1*x2)" value={values.w} id="w" x={200} y={150} grad={grads.w} />
                <Node label="y (sin)" value={values.y} id="y" x={320} y={150} grad={grads.y} />

                {mode === 'reverse' && step >= 1 && (
                    <text x="260" y="140" fill="#ef4444" fontSize="10">cos(w)</text>
                )}
            </svg>

            <div style={{ marginTop: '10px', fontSize: '12px', color: '#94a3b8' }}>
                Inputs: x1={inputs.x1}, x2={inputs.x2}
            </div>
        </div>
    );
};

const domContainer = document.querySelector('#ad-animation-container');
ReactDOM.render(<ADAnimation />, domContainer);
