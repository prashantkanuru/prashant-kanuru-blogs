const PytorchPointerAnimation = () => {
    const [step, setStep] = React.useState(0);

    // Scenario: z = x * y
    // x, y are leaf tensors (requires_grad=True)
    // z is the product.
    // z.grad_fn is MulBackward0

    const steps = [
        {
            text: "Initial State: Tensors created. x and y are leaf nodes.",
            highlight: ['tensor_x', 'tensor_y']
        },
        {
            text: "Forward Pass: z = x * y. PyTorch creates a 'MulBackward' Node.",
            highlight: ['node_mul', 'tensor_z']
        },
        {
            text: "Dynamic Graph: z.grad_fn points to MulBackward Node.",
            highlight: ['ptr_grad_fn'],
            arrow: 'grad_fn_link'
        },
        {
            text: "Context (ctx): MulBackward saves 'x' and 'y' (Primals) for backward pass.",
            highlight: ['ctx', 'saved_tensors']
        },
        {
            text: "The Tape: MulBackward adds x and y (accumulators) to 'next_functions'.",
            highlight: ['next_functions'],
            arrow: 'next_fn_links'
        },
        {
            text: "Backward Call: z.backward() triggers execution starting at MulBackward.",
            highlight: ['node_mul'],
            action: 'executing'
        },
        {
            text: "Execution: ctx is accessed to retrieve x and y to compute gradients.",
            highlight: ['ctx', 'saved_tensors'],
            action: 'reading'
        },
        {
            text: "Propagation: Gradients flow down 'next_functions' to updating x.grad and y.grad.",
            highlight: ['accum_x', 'accum_y'],
            arrow: 'flow_down'
        }
    ];

    const handleNext = () => setStep(Math.min(step + 1, steps.length - 1));
    const handlePrev = () => setStep(Math.max(step - 1, 0));

    const Box = ({ x, y, width, height, label, color, id }) => {
        const isActive = steps[step].highlight.includes(id);
        return (
            <g>
                <rect x={x} y={y} width={width} height={height} rx="5" fill={isActive ? color : '#334155'} stroke={isActive ? '#cbd5e1' : 'none'} strokeWidth="2" />
                <text x={x + width / 2} y={y + 20} textAnchor="middle" fill="#fff" fontSize="12" fontWeight="bold">{label}</text>
            </g>
        );
    };

    return (
        <div style={{ fontFamily: 'Inter, sans-serif', background: 'rgba(30, 41, 59, 0.5)', padding: '20px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.1)', color: '#f8fafc', marginTop: '20px' }}>
            <h3 style={{ marginTop: 0, color: '#a855f7' }}>PyTorch Internals: The Tape & ctx</h3>

            <div style={{ marginBottom: '15px', height: '40px' }}>
                {steps[step].text}
            </div>

            <div style={{ marginBottom: '20px' }}>
                <button onClick={handlePrev} disabled={step === 0} style={{ marginRight: '10px', padding: '5px 10px', borderRadius: '4px', border: 'none', cursor: 'pointer', background: '#475569', color: 'white', opacity: step === 0 ? 0.5 : 1 }}>Prev</button>
                <button onClick={handleNext} disabled={step === steps.length - 1} style={{ padding: '5px 10px', borderRadius: '4px', border: 'none', cursor: 'pointer', background: '#6366f1', color: 'white', opacity: step === steps.length - 1 ? 0.5 : 1 }}>Next</button>
            </div>

            <svg viewBox="0 0 500 300" style={{ width: '100%', height: '300px', background: '#0f172a', borderRadius: '8px' }}>

                {/* Tensors */}
                <Box x={50} y={50} width={80} height={40} label="Tensor x" color="#0ea5e9" id="tensor_x" />
                <Box x={150} y={50} width={80} height={40} label="Tensor y" color="#0ea5e9" id="tensor_y" />

                {/* Output Tensor z */}
                {(step >= 1) && (
                    <Box x={350} y={50} width={80} height={40} label="Tensor z" color="#0ea5e9" id="tensor_z" />
                )}

                {/* MulBackward Node */}
                {(step >= 1) && (
                    <g>
                        <rect x={320} y={120} width={140} height={150} rx="8" fill={steps[step].highlight.includes('node_mul') ? '#6366f1' : '#1e293b'} stroke="#6366f1" strokeWidth="2" />
                        <text x={390} y={140} textAnchor="middle" fill="#fff" fontSize="14" fontWeight="bold">Node (MulBackward)</text>

                        {/* grad_fn pointer */}
                        <line x1="390" y1="90" x2="390" y2="120" stroke="#cbd5e1" strokeWidth="2" strokeDasharray="4" markerEnd="url(#arrow)" />
                        <text x={400} y={110} fill="#94a3b8" fontSize="10">.grad_fn</text>

                        {/* ctx */}
                        <g transform="translate(330, 150)">
                            <rect width={120} height={50} fill={steps[step].highlight.includes('ctx') ? '#be185d' : '#334155'} rx="4" />
                            <text x={10} y={20} fill="#fff" fontSize="12" fontWeight="bold">ctx (Context)</text>
                            <text x={10} y={40} fill="#e2e8f0" fontSize="10">Saved: [x, y]</text>
                        </g>

                        {/* next_functions */}
                        <g transform="translate(330, 210)">
                            <rect width={120} height={50} fill={steps[step].highlight.includes('next_functions') ? '#fbbf24' : '#334155'} rx="4" />
                            <text x={10} y={20} fill={(steps[step].highlight.includes('next_functions') ? '#000' : '#fff')} fontSize="12" fontWeight="bold">next_functions</text>
                            <text x={10} y={40} fill={(steps[step].highlight.includes('next_functions') ? '#000' : '#e2e8f0')} fontSize="10">List of Edges</text>
                        </g>
                    </g>
                )}

                {/* Accumulate Grad Nodes (Implicit) */}
                {(step >= 4) && (
                    <g>
                        {/* Edge to x */}
                        <path d="M 330 235 C 200 235, 200 100, 90 90" stroke="#fbbf24" strokeWidth="2" fill="none" markerEnd="url(#arrow)" strokeDasharray={step >= 7 ? "0" : "5,5"} />
                        {/* Edge to y */}
                        <path d="M 330 235 C 250 235, 250 100, 190 90" stroke="#fbbf24" strokeWidth="2" fill="none" markerEnd="url(#arrow)" strokeDasharray={step >= 7 ? "0" : "5,5"} />
                    </g>
                )}

                <defs>
                    <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                        <path d="M0,0 L0,6 L9,3 z" fill="#cbd5e1" />
                    </marker>
                </defs>

            </svg>
        </div>
    );
};

const domContainerPytorch = document.querySelector('#pytorch-pointer-animation-container');
ReactDOM.render(<PytorchPointerAnimation />, domContainerPytorch);
