"""Interactive graph experiment, kept separate from the legacy CNN workflow."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from qiskit import QuantumCircuit, qasm2

from src.circuit_utils import check_circuit_equivalence
from src.graph_optimizer import optimize_graph, dependency_edges
from src.gnn_optimizer import load_gnn
from src.experiment_data import make_family
from scripts.benchmark import circuit_metrics

st.set_page_config(page_title='Graph Optimizer Lab', page_icon='⚛️', layout='wide')
st.title('Graph Optimizer Lab')
st.write('Compare dependency-aware rewrites with an experimental learned selector.')
st.caption('Standard bound gates only. Numerical verification supports up to 8 qubits. '
           'This experiment does not establish an advantage over existing compilers.')

family = st.selectbox('Example circuit', ['interleaved', 'rotations', 'entangling', 'adjacent'])
source = st.text_area('OpenQASM 2 circuit', qasm2.dumps(make_family(family, 7, blocks=3)), height=280,
                      key=f'qasm_{family}')
checkpoint = ROOT / 'artifacts/gnn.pt'
options = ['Graph greedy'] + (['GNN (experimental)'] if checkpoint.exists() else [])
strategy = st.selectbox('Rewrite selection', options)
if not checkpoint.exists():
    st.info('Train the GNN with: python -m scripts.train_gnn')
budget = st.slider('Maximum rewrite steps', 1, 500, 100)

if st.button('Optimise and verify', type='primary'):
    try:
        original = QuantumCircuit.from_qasm_str(source)
        if original.num_qubits > 8:
            raise ValueError('This interactive experiment supports at most 8 qubits.')
        selector = None
        if strategy.startswith('GNN'):
            model, metadata = load_gnn(checkpoint)
            selector = model.select
        candidate, trace = optimize_graph(original, selector, budget)
        verification = check_circuit_equivalence(original, candidate)
        final = candidate if verification['equivalent'] else original
        if verification['equivalent']:
            st.success('Equivalent up to global phase within numerical tolerance.')
        else:
            st.warning(f"{verification['status']}: {verification['reason']} Original retained.")
        st.table({'Metric': ['Gates', 'Two-qubit gates', 'Depth'],
                  'Original': list(circuit_metrics(original).values()),
                  'Final': list(circuit_metrics(final).values())})
        left, right = st.columns(2)
        left.code(str(original.draw(output='text')))
        right.code(str(final.draw(output='text')))
        st.write(f'{len(dependency_edges(original))} wire dependencies; {len(trace)} rewrites applied.')
        st.dataframe(trace)
        report = {'strategy': strategy, 'trace': trace, 'verification': verification,
                  'original_qasm': qasm2.dumps(original), 'final_qasm': qasm2.dumps(final)}
        st.download_button('Download experiment report', json.dumps(report, indent=2),
                           'graph-experiment.json', 'application/json')
    except Exception as error:
        st.error(str(error))
