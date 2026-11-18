from functools import partial

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from spendguard.graph.config import GraphConfig
from spendguard.graph.nodes.execute import execute
from spendguard.graph.nodes.finalize import finalize
from spendguard.graph.nodes.human_review import human_review
from spendguard.graph.nodes.intake import intake
from spendguard.graph.nodes.plan import plan
from spendguard.graph.nodes.retrieve_policy import retrieve_policy
from spendguard.graph.nodes.route import route
from spendguard.graph.nodes.run_judgement_checks import run_judgement_checks
from spendguard.graph.nodes.run_rule_checks import run_rule_checks
from spendguard.graph.nodes.supply_whole_policy import supply_whole_policy
from spendguard.graph.nodes.verify import verify
from spendguard.graph.state import ReviewState
from spendguard.rules.decisions import Route

ReviewGraph = CompiledStateGraph[ReviewState, None, ReviewState, ReviewState]
NODES = (
    intake,
    run_rule_checks,
    plan,
    run_judgement_checks,
    verify,
    route,
    human_review,
    execute,
    finalize,
)


def _next_after_route(state: ReviewState) -> Route:
    return state['route']


def build_graph(config: GraphConfig) -> ReviewGraph:
    builder: StateGraph[ReviewState, None, ReviewState, ReviewState] = StateGraph(ReviewState)
    policy_node = {'retrieved_clauses': retrieve_policy, 'whole_policy': supply_whole_policy}[
        config.policy_context
    ]
    for node in (*NODES, policy_node):
        builder.add_node(node.__name__, partial(node, graph_config=config))
    builder.add_edge(START, 'intake')
    builder.add_edge('intake', 'run_rule_checks')
    builder.add_edge('intake', 'plan')
    builder.add_edge('plan', policy_node.__name__)
    builder.add_edge(policy_node.__name__, 'run_judgement_checks')
    builder.add_edge(['run_rule_checks', 'run_judgement_checks'], 'verify')
    builder.add_edge('verify', 'route')
    builder.add_conditional_edges(
        'route', _next_after_route, {'auto_approve': 'execute', 'human_review': 'human_review'}
    )
    builder.add_edge('human_review', 'execute')
    builder.add_edge('execute', 'finalize')
    builder.add_edge('finalize', END)
    return builder.compile(checkpointer=config.checkpointer)
