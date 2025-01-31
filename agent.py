import autogen
from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv
import os


load_dotenv(".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

config = [{
    'model': 'gpt-4o',
    'api_key': OPENAI_API_KEY
}]

llm_config={"config_list": config}


task = '''
 **Task**: As an architect, you are required to design a solution for the
 following business requirements:
    - Data storage for massive amounts of pdf files
    - Real-time data analytics and machine learning pipeline
    - Scalability
    - Cost Optimization
    - Region pairs in Europe, for disaster recovery
    - Tools for monitoring and observability
    - Timeline: 6 months

    Break down the problem using a Chain-of-Thought approach. Ensure that your
    solution architecture is following best practices.
    '''

cloud_prompt = '''
**Role**: You are an expert cloud architect. You need to develop architecture proposals
using either cloud-specific PaaS services, or cloud-agnostic ones.
The final proposal should consider all 3 main cloud providers: Azure, AWS and GCP, and provide
a data architecture for each. At the end, briefly state the advantages of cloud over on-premises
architectures, and summarize your solutions for each cloud provider using a table for clarity.
'''
cloud_prompt += task

oss_prompt = '''
**Role**: You are an expert on-premises, open-source software architect. You need
to develop architecture proposals without considering cloud solutions.
 Only use open-source frameworks that are popular and have lots of active contributors.
 At the end, briefly state the advantages of open-source adoption, and summarize your
 solutions using a table for clarity.
'''
oss_prompt += task

lead_prompt = '''
**Role**: You are a lead Architect tasked with managing a conversation between
the cloud and the open-source Architects.
Each Architect will perform a task and respond with their resuls. You will critically
review those and also ask for, or point to, the disadvantages of their solutions.
You will review each result, and choose the best solution in accordance with the business
requirements and architecture best practices. You will use any number of summary tables to
communicate your decision.
'''
lead_prompt += task

user_proxy=UserProxyAgent(
    name="supervisor",
    system_message = "A Human Head of Architecture",
    code_execution_config={
        "last_n_messages": 2,
        "work_dir": "groupchat",
        "use_docker": False,
    },
    human_input_mode="NEVER",


)

cloud_agent = AssistantAgent(
    name = "cloud",
    system_message = cloud_prompt,
    llm_config={"config_list": config}
)

oss_agent = AssistantAgent(
    name = "oss",
    system_message = oss_prompt,
    llm_config={"config_list": config}
)

lead_agent = AssistantAgent(
    name = "lead",
    system_message = lead_prompt,
    llm_config={"config_list": config}
)


def state_transition(last_speaker,groupchat):
    messages = groupchat.messages

    if last_speaker is user_proxy:
        return cloud_agent
    elif last_speaker is cloud_agent:
        return oss_agent
    elif last_speaker is oss_agent:
        return lead_agent
    elif last_speaker is lead_agent:
        # lead -> end
        return None


groupChat = autogen.GroupChat(
    agents=[user_proxy, cloud_agent, oss_agent, lead_agent],
    messages=[],
    max_round=6,
    speaker_selection_method=state_transition,
)

manager = autogen.GroupChatManager(groupchat=groupChat, llm_config=llm_config)


def initiate_chat(user_message: str):
    return user_proxy.initiate_chat(
        manager, message=user_message
    ).summary





