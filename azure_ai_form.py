import os
import re
import json
import sys
import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from rich.console import Console
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
import argparse

# Attempt to load environment variables from .env file
# This should be one of the first things to do
try:
    from dotenv import load_dotenv
    load_dotenv() # Loads variables from .env into environment
    # print("DEBUG: .env file loaded successfully by dotenv.") # Optional: for debugging
except ImportError:
    # dotenv is not installed, proceed without it (rely on system env vars)
    # print("DEBUG: dotenv not found, relying on system environment variables.") # Optional: for debugging
    pass

# Initialize console for rich text output
console = Console()

# Try to import voice input utility
try:
    from utils.voice_input import record_and_transcribe, MLX_WHISPER_AVAILABLE
    VOICE_AVAILABLE = True
    console.print("[dim]Voice input module loaded.[/dim]")
except ImportError:
    VOICE_AVAILABLE = False
    MLX_WHISPER_AVAILABLE = False # Ensure this is also False if import fails
    console.print("[yellow]Warning: utils.voice_input module not found. Voice input will be disabled.[/yellow]")
    # Define a placeholder if needed, or just rely on VOICE_AVAILABLE flag
    def record_and_transcribe(*args, **kwargs):
        console.print("[bold red]Voice input functionality not available.[/bold red]")
        return None

# Azure AI Projects SDK imports
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import CodeInterpreterTool
from azure.identity import DefaultAzureCredential

# Available Azure OpenAI models
AZURE_MODELS = {
    "gpt-4o": {"description": "More powerful model"},
    "gpt-4o-mini": {"description": "More affordable model"}
}
DEFAULT_MODEL = "gpt-4o-mini"

# Connection string for Azure AI Project (from main.py)
AZURE_AI_CONN_STR = os.getenv("AZURE_AI_CONN_STR", "")

# Core data models
@dataclass
class Question:
    """Represents a question in the questionnaire."""
    text: str
    index: int  # Position in the original list

@dataclass
class Section:
    """Represents a section in the questionnaire."""
    title: str
    questions: List[Question]

@dataclass
class SectionFeedback:
    """Feedback on the completion status of a section."""
    content: str  # The content user provided for the section
    is_complete: bool  # Whether all questions have been answered sufficiently
    missing_questions: List[int]  # Indices of questions not yet fully answered
    completion_percentage: float  # Estimated percentage of completion (0-100)

@dataclass
class SynthesizedResponse:
    """A synthesized answer to a question based on the conversation."""
    question: str  # The original question from the questionnaire
    answer: str  # A synthesized answer to the question based on the conversation

@dataclass
class QuestionnaireResult:
    """The final result of processing a questionnaire."""
    conversation: Dict[str, str]  # Section title -> conversation content
    synthesized_answers: Dict[str, List[Dict[str, str]]]  # Section title -> list of Q&A pairs

def check_connection_string(conn_str: str) -> bool:
    """
    Check that the connection string is valid.
    
    Args:
        conn_str: The connection string to check
        
    Returns:
        bool: True if the connection string is valid, False otherwise
    """
    # Simple validation - make sure it has 4 parts separated by semicolons
    parts = conn_str.split(';')
    if len(parts) != 4:
        console.print("[bold red]Error: Connection string must have 4 parts separated by semicolons.[/bold red]")
        console.print("[yellow]Format: <HostName>;<AzureSubscriptionId>;<ResourceGroup>;<ProjectName>[/yellow]")
        return False
    
    # Display current configuration (masked for privacy)
    host = parts[0]
    subscription = parts[1]
    resource_group = parts[2]
    project_name = parts[3]
    
    masked_subscription = subscription[:8] + "..." + subscription[-4:] if len(subscription) > 12 else subscription
    
    console.print(f"[dim]Azure AI Host: {host}[/dim]")
    console.print(f"[dim]Azure Subscription ID: {masked_subscription}[/dim]")
    console.print(f"[dim]Azure Resource Group: {resource_group}[/dim]")
    console.print(f"[dim]Azure Project Name: {project_name}[/dim]")
    
    return True

def create_ai_project_client(conn_str: str = AZURE_AI_CONN_STR) -> Optional[AIProjectClient]:
    """
    Create and return an Azure AI Project client.
    
    Args:
        conn_str: The connection string for the Azure AI Project
        
    Returns:
        Optional[AIProjectClient]: The client if successful, None otherwise
    """
    try:
        # Create the client
        console.print("[dim]Connecting to Azure AI Project...[/dim]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Connecting to Azure AI Project...[/bold blue]"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Connecting", total=None)
            client = AIProjectClient.from_connection_string(
                credential=DefaultAzureCredential(), 
                conn_str=conn_str
            )
            progress.update(task, completed=True)
        
        console.print("[green]✓ Connected to Azure AI Project[/green]")
        return client
    except Exception as e:
        console.print(f"[bold red]✗ Failed to connect to Azure AI Project: {str(e)}[/bold red]")
        console.print("[yellow]Please check your connection string and try again.[/yellow]")
        return None

def parse_markdown_sections(markdown_content: str) -> Tuple[str, List[Section]]:
    """
    Parse a markdown file with sections and questions.
    
    Args:
        markdown_content: The content of the markdown file
        
    Returns:
        Tuple[str, List[Section]]: (preamble_content, sections_found)
    """
    # Find all sections
    section_pattern = r"(?:^|\n)(?:##|###)\s*(.+?)\n([\s\S]*?)(?=(?:^|\n)(?:##|###)|$)"
    
    # Try to find the first section match to identify preamble
    first_section_match = re.search(section_pattern, markdown_content)
    preamble_content = ""
    content_to_parse_for_sections = markdown_content

    if first_section_match:
        preamble_end_index = first_section_match.start()
        preamble_content = markdown_content[:preamble_end_index].strip()
        content_to_parse_for_sections = markdown_content[preamble_end_index:]
    else:
        # No ## or ### sections found, the whole content might be preamble (e.g. only a # title)
        # or it might be content without any specific section markers we look for.
        # For now, if no ## or ###, treat all as preamble if it contains a # header, otherwise assume no sections.
        if markdown_content.strip().startswith("#"):
            preamble_content = markdown_content.strip()
            # No sections will be found by the loop below if this is the case
        # else: sections_found remains empty, preamble is empty. Handled by existing logic.

    sections_found = []
    for title, body in re.findall(section_pattern, content_to_parse_for_sections):
        questions = []
        
        # Look for questions with numbered patterns like "**1.**", "1.", etc.
        # Updated pattern to handle special characters and more formats
        question_pattern = r"(?:\*\*)?(\d+)(?:\.\*\*|\.)[ \t]+(.*?)(?=\n\n|\n(?:\*\*)?(?:\d+)(?:\.\*\*|\.)|\n---|\n$|$)"
        
        for idx, match in enumerate(re.finditer(question_pattern, body)):
            q_number = match.group(1)
            q_text = match.group(2).strip()
            
            # Clean up any markdown formatting and whitespace
            q_text = re.sub(r'\*\*|\*', '', q_text)  # Remove bold/italic
            q_text = re.sub(r'\s+', ' ', q_text).strip()  # Normalize whitespace
            q_text = q_text.replace('–', '-')  # Replace en dash with regular hyphen
            q_text = q_text.replace('—', '-')  # Replace em dash with regular hyphen
            
            if q_text:
                questions.append(Question(text=q_text, index=idx))
        
        # Fallback: If no questions found using the pattern, try line by line
        if not questions:
            for idx, line in enumerate(body.splitlines()):
                line = line.strip()
                # Look for lines that start with a number followed by a period
                if re.match(r'^(?:\*\*)?(\d+)(?:\.\*\*|\.)[ \t]+', line):
                    # Extract the question text
                    q_text = re.sub(r'^(?:\*\*)?(\d+)(?:\.\*\*|\.)[ \t]+', '', line).strip()
                    q_text = re.sub(r'\*\*|\*', '', q_text)  # Remove bold/italic
                    q_text = q_text.replace('–', '-')  # Replace en dash with regular hyphen
                    q_text = q_text.replace('—', '-')  # Replace em dash with regular hyphen
                    if q_text:
                        questions.append(Question(text=q_text, index=idx))
        
        sections_found.append(Section(title=title.strip(), questions=questions))
        
        # Debug output
        console.print(f"[dim]Found {len(questions)} questions in section '{title}'[/dim]")
        if questions:
            console.print(f"[dim]First question: '{questions[0].text}'[/dim]")
            
    return preamble_content, sections_found

def parse_unfinished_qa(unfinished_content: Optional[str]) -> Dict[str, str]:
    """
    Parse the unfinished Q&A file into a structured format.
    
    Args:
        unfinished_content: The content of the unfinished Q&A file, or None
        
    Returns:
        Dict[str, str]: Section title -> conversation content
    """
    if not unfinished_content:
        return {}
        
    sections = {}
    current_section = None
    current_content = []
    
    # Simple parsing logic for Q&A markdown file
    for line in unfinished_content.splitlines():
        # Check for section headers
        if line.startswith('## '):
            # Save previous section if exists
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content)
                current_content = []
            
            # Start new section
            current_section = line[3:].strip()
        elif current_section:
            # Add content to current section
            current_content.append(line)
    
    # Save last section
    if current_section and current_content:
        sections[current_section] = '\n'.join(current_content)
        
    return sections

class AzureAIQuestionnaire:
    """Main class for handling the questionnaire workflow with Azure AI."""
    
    def __init__(self, model_id: str = DEFAULT_MODEL, conn_str: str = AZURE_AI_CONN_STR):
        """
        Initialize the questionnaire handler.
        
        Args:
            model_id: The model ID to use for all agents
            conn_str: The connection string for the Azure AI Project
        """
        self.model_id = model_id
        self.conn_str = conn_str
        self.client = None
        self.agents = {}
        self.threads = {}
        self.max_turns_per_section = 5
        self.completion_threshold = 90  # Require at least 90% completion to finish a section
        self.markdown_input = None
        self.unfinished_content = None
        self.temp_results_file = "temp_results.md"  # Name of the temporary results file
        self._parsed_sections = None  # Cache for parsed sections
        self._preamble_content = "" # To store the content before the first ## or ### section
        
    def setup(self) -> bool:
        """
        Set up the Azure AI client and create agents.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        if not check_connection_string(self.conn_str):
            return False
            
        self.client = create_ai_project_client(self.conn_str)
        if not self.client:
            return False
            
        return self._create_agents()
    
    def _create_agents(self) -> bool:
        """
        Create the necessary agents for the questionnaire.
        
        Returns:
            bool: True if all agents were created successfully, False otherwise
        """
        try:
            console.print(f"[green]Creating agents with model: {self.model_id}[/green]")
            
            # Create evaluator agent
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Creating evaluator agent...[/bold blue]"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("Creating", total=None)
                
                self.agents["evaluator"] = self.client.agents.create_agent(
                    model=self.model_id,
                    name="Section Evaluator",
                    instructions="""You are a meticulous section evaluator. Your SOLE task is to evaluate conversation completeness.
                    You MUST provide your response ONLY as a single, valid JSON object, with no other text before or after it.
                    The JSON object must contain these exact fields:
                    - "content": A brief summary or the raw conversation content you evaluated.
                    - "is_complete": A boolean (true/false) indicating if all questions/topics were fully addressed.
                    - "missing_questions": A list of 0-based integer indices for questions/topics not fully answered. If all are complete, this MUST be an empty list [].
                    - "completion_percentage": A float (0.0 to 100.0) estimating overall completion.

                    Example of a valid JSON response:
                    {
                        "content": "User discussed their name and preferred nickname.",
                        "is_complete": false,
                        "missing_questions": [1],
                        "completion_percentage": 50.0
                    }
                    Example if complete:
                    {
                        "content": "User fully answered all questions about their background and experience.",
                        "is_complete": true,
                        "missing_questions": [],
                        "completion_percentage": 100.0
                    }
                    Ensure your entire response is ONLY this JSON structure."""
                )
                progress.update(task, completed=True)
            
            console.print(f"[green]✓ Created evaluator agent (ID: {self.agents['evaluator'].id})[/green]")
            
            # Create interviewer agent
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Creating interviewer agent...[/bold blue]"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("Creating", total=None)
                
                self.agents["interviewer"] = self.client.agents.create_agent(
                    model=self.model_id,
                    name="Interviewer",
                    instructions="""You are an interviewer conducting a conversation to gather information.
                    Your goal is to craft simple, conversational questions that are easy to answer.
                    Focus on asking one clear question at a time rather than overwhelming with multiple questions.
                    Make your questions friendly and inviting, not formal or intimidating.
                    When you receive a follow-up topic, ask a natural question about it that builds on the conversation.
                    """
                )
                progress.update(task, completed=True)
            
            console.print(f"[green]✓ Created interviewer agent (ID: {self.agents['interviewer'].id})[/green]")
            
            # Create synthesizer agent
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Creating synthesizer agent...[/bold blue]"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("Creating", total=None)
                
                self.agents["synthesizer"] = self.client.agents.create_agent(
                    model=self.model_id,
                    name="Response Synthesizer",
                    instructions="""You are a response synthesizer.
                    Your task is to analyze a conversation transcript and extract a coherent, comprehensive answer
                    to a specific question based on the information in the conversation.
                    
                    For each question:
                    1. Find all relevant information in the transcript
                    2. Synthesize this into a clear, concise answer that directly addresses the question
                    3. Use only information from the conversation
                    
                    Format your response as valid JSON with these fields:
                    {
                        "question": "The original question",
                        "answer": "The synthesized answer"
                    }
                    """
                )
                progress.update(task, completed=True)
            
            console.print(f"[green]✓ Created synthesizer agent (ID: {self.agents['synthesizer'].id})[/green]")
            
            # Create threads for each agent
            self._create_threads()
            
            return True
            
        except Exception as e:
            console.print(f"[bold red]Error creating agents: {str(e)}[/bold red]")
            return False
    
    def _create_threads(self):
        """Create threads for each agent."""
        for agent_name in self.agents:
            self.threads[agent_name] = self.client.agents.create_thread()
            console.print(f"[dim]Created thread for {agent_name} (ID: {self.threads[agent_name].id})[/dim]")
            
    def run_agent(self, agent_name: str, prompt: str) -> Tuple[bool, Any]:
        """
        Run an agent with a prompt and return the response.
        
        Args:
            agent_name: The name of the agent to run
            prompt: The prompt to send to the agent
            
        Returns:
            Tuple[bool, Any]: (success, response)
        """
        try:
            # Get the agent and thread
            agent = self.agents.get(agent_name)
            thread = self.threads.get(agent_name)
            
            if not agent or not thread:
                console.print(f"[bold red]Agent or thread not found for {agent_name}[/bold red]")
                return False, None
                
            # Create a message with the prompt
            message = self.client.agents.create_message(
                thread_id=thread.id,
                role="user",
                content=prompt,
            )
            
            # Run the agent
            run = self.client.agents.create_and_process_run(
                thread_id=thread.id, 
                agent_id=agent.id
            )
            
            if run.status == "failed":
                console.print(f"[bold red]Run failed: {run.last_error}[/bold red]")
                return False, None
                
            # Get the response message
            messages = self.client.agents.list_messages(thread_id=thread.id)
            last_msg = messages.get_last_text_message_by_role("assistant")
            
            if not last_msg:
                console.print(f"[bold red]No response from {agent_name}[/bold red]")
                return False, None
                
            return True, last_msg.text.value
            
        except Exception as e:
            console.print(f"[bold red]Error running agent {agent_name}: {str(e)}[/bold red]")
            return False, None
    
    def set_input(self, markdown_content: str, max_turns_per_section: int = 5, unfinished_qa: Optional[str] = None) -> 'AzureAIQuestionnaire':
        """
        Set the input content for the questionnaire.
        
        Args:
            markdown_content: The markdown content with sections and questions
            max_turns_per_section: Maximum number of conversation turns per section
            unfinished_qa: Optional content from an unfinished Q&A file
            
        Returns:
            self: For method chaining
        """
        self.markdown_input = markdown_content
        self.max_turns_per_section = max_turns_per_section
        self.unfinished_content = unfinished_qa
        # Parse sections once and cache them
        self._preamble_content, self._parsed_sections = parse_markdown_sections(markdown_content)
        return self
    
    def process_questionnaire(self) -> QuestionnaireResult:
        """
        Process the questionnaire with the agent's assistance.
        
        Returns:
            QuestionnaireResult: The conversation transcripts and synthesized answers
        """
        if not self.markdown_input:
            raise ValueError("No markdown input provided. Call set_input() first.")
            
        # Use cached sections instead of parsing again
        sections = self._parsed_sections
        if not sections and not self._preamble_content:
            # If parsing found no sections and no preamble, it implies the input was empty or not parsable as expected.
            # We might want to handle this, e.g., by raising an error or returning an empty result.
            # For now, let it proceed, it will likely result in an empty output file, which is acceptable.
            console.print("[yellow]Warning: No sections or preamble found in the input markdown.[/yellow]")

        unfinished_sections = parse_unfinished_qa(self.unfinished_content)
        
        all_responses = {}
        final_synthesized_answers = {}
        
        if unfinished_sections:
            console.print(f"[bold green]Found {len(unfinished_sections)} sections with unfinished answers[/bold green]")
        
        # Display command help at the beginning
        console.print("\n[bold blue]Available commands during conversation:[/bold blue]")
        console.print("  - Type [bold green]exit[/bold green] or [bold green]quit[/bold green] to end the session and save results")
        console.print("  - Type [bold green]voice[/bold green] or [bold green]v[/bold green] to use voice input (if available)")
        console.print("  - Or simply type your response as normal\n")
        
        console.print(f"[dim]A temporary results file will be updated after each section: {self.temp_results_file}[/dim]")
        
        # Initialize the temporary results file once at the start
        self._update_temp_results(all_responses, final_synthesized_answers)
        
        # Process each section
        for section in sections:
            title = section.title
            questions = section.questions
            
            # Skip sections with no questions
            if not questions:
                console.print(f"\n[blue]=== Section: {title} ===[/blue]")
                console.print("[yellow]No specific questions found for this section. Moving to next section.[/yellow]")
                continue
            
            # Process this section
            success, section_responses, section_answers = self._process_section(title, questions, unfinished_sections.get(title))
            
            # Store results
            if success:
                all_responses[title] = section_responses
                final_synthesized_answers[title] = section_answers
                
                # Update the temporary results file after each successful section
                self._update_temp_results(all_responses, final_synthesized_answers, title)
            else:
                # Exit early if requested
                console.print("[bold yellow]Exiting section processing at user request[/bold yellow]")
                # Still update the temporary results with what we have
                self._update_temp_results(all_responses, final_synthesized_answers, title)
                break
                
        # Create and return the final result
        result = QuestionnaireResult(
            conversation=all_responses,
            synthesized_answers=final_synthesized_answers
        )
        
        # Save the final results
        output_file = self._save_results(result)
        
        # Delete the temporary file since we now have the final file
        self._delete_temp_results()
        
        return result
    
    def _get_user_input(self, prompt_text: str) -> str:
        """
        Get user input with support for commands.
        
        Args:
            prompt_text: The text to display when prompting
            
        Returns:
            str: The user input or command
        """
        response = Prompt.ask(prompt_text)
        
        # Support for commands
        if response.lower() in ["exit", "quit", "x", "q"]:
            console.print(f"[dim]Recognized '{response}' as exit command[/dim]")
            return "exit"
        elif response.lower() in ["voice", "v"]:
            if VOICE_AVAILABLE:
                console.print("[bold blue]🎤 Voice input activated. Recording...[/bold blue]")
                # Determine transcription mode
                # Default to local if mlx_whisper is available, otherwise azure
                transcription_mode = "local" if MLX_WHISPER_AVAILABLE else "azure"
                
                if transcription_mode == "azure":
                    # Check for Azure OpenAI credentials for voice if not already handled by voice_input.py
                    if not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"):
                        console.print("[bold red]Azure OpenAI endpoint and API key are required for voice transcription in Azure mode.[/bold red]")
                        console.print("[yellow]Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables.[/yellow]")
                        console.print("[yellow]Voice input disabled for this attempt. Please type your response.[/yellow]")
                        return self._get_user_input(prompt_text) # Re-prompt for text input

                transcribed_text = record_and_transcribe(
                    output_file="temp_voice_input.wav", # Temporary file for voice recording
                    mode=transcription_mode,
                    # mlx_model can be configured if needed, uses default from voice_input.py
                )
                if transcribed_text:
                    console.print(f"[green]Transcription:[/green] {transcribed_text}")
                    return transcribed_text
                else:
                    console.print("[yellow]Voice input failed or no transcription received. Please type your response.[/yellow]")
                    return self._get_user_input(prompt_text) # Re-prompt for text input
            else:
                console.print("[yellow]Voice input module not available. Please type your response.[/yellow]")
                return self._get_user_input(prompt_text) # Re-prompt for text input
            
        return response

    def _process_section(self, title: str, questions: List[Question], existing_content: Optional[str] = None) -> Tuple[bool, str, List[Dict[str, str]]]:
        """
        Process a single section of the questionnaire.
        
        Args:
            title: The section title
            questions: The list of questions for this section
            existing_content: Optional existing content for this section
            
        Returns:
            Tuple[bool, str, List[Dict[str, str]]]: (success, response_content, answers)
        """
        console.print(f"\n[blue]=== Section: {title} ===[/blue]")
        
        # Initialize with existing content if provided
        interview_content = ""
        if existing_content:
            console.print(f"[bold blue]Using existing unfinished content for this section[/bold blue]")
            interview_content = existing_content
            console.print("\n[dim]Existing content:[/dim]")
            console.print(f"[dim]{interview_content}[/dim]")
            console.print("[green]Automatically using existing content as a starting point[/green]")
        
        # If no existing content, start fresh conversation
        if not interview_content:
            # Initial simple prompt
            init_text = (
                f"You are interviewing someone about '{title}'. "
                f"Start the conversation with a single, simple opening question that's easy to answer. "
                f"The goal is to eventually cover these topics: {', '.join(q.text for q in questions)}. "
                f"But begin with just ONE friendly, conversational question to get them started."
            )
            
            # Show that the agent is working
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Preparing initial question...[/bold blue]"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("Preparing", total=None)
                success, init_prompt = self.run_agent("interviewer", init_text)
                progress.update(task, completed=True)
                
            if not success:
                console.print("[bold red]Failed to get initial question. Skipping section.[/bold red]")
                return False, "", []
                
            console.print(init_prompt)
            
            # Initial response with command support
            response = self._get_user_input("[bold magenta]>> Your response[/bold magenta]")
            
            # Check for exit commands
            if response.lower() == "exit":
                # Save partial results and exit
                console.print("\n[bold yellow]Exiting section at user request[/bold yellow]")
                return False, interview_content, []
            
            interview_content = response + "\n"
        
        # Track progress
        current_progress = 0
        
        # Iterative conversation
        for turn in range(self.max_turns_per_section):
            # Evaluate coverage so far
            eval_text = (
                f"TOPICS TO COVER:\n"
                f"{', '.join(q.text for q in questions)}\n\n"
                f"CONVERSATION SO FAR:\n{interview_content}\n\n"
                f"Evaluate if the conversation has addressed all topics completely. "
                f"If not, indicate missing topics by their 0-based index. "
                f"Also estimate a completion_percentage (0-100) of how completely the topics have been covered."
            )
            
            try:
                # Show that the agent is working
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]Evaluating responses...[/bold blue]"),
                    console=console,
                    transient=True,
                ) as progress:
                    task = progress.add_task("Evaluating", total=None)
                    success, eval_response = self.run_agent("evaluator", eval_text)
                    progress.update(task, completed=True)
                
                if not success:
                    console.print("[bold red]Failed to evaluate responses. Continuing with next question.[/bold red]")
                    # If evaluation fails, just pick a random question
                    if questions:
                        next_topic_idx = turn % len(questions)
                        next_topic = questions[next_topic_idx].text
                    else:
                        next_topic = "the topic"
                else:
                    # Parse the evaluation response
                    try:
                        feedback = self._parse_feedback(eval_response)
                        
                        # Track completion percentage
                        if hasattr(feedback, 'completion_percentage'):
                            current_progress = feedback.completion_percentage
                            console.print(f"\n📊 [bold green]Progress: {int(current_progress)}% complete[/bold green]")
                        
                        # Check if section is complete
                        section_complete = (current_progress >= self.completion_threshold and 
                                           (feedback.is_complete or not feedback.missing_questions))
                        
                        if section_complete:
                            console.print(f"✅ [bold green]Section complete! ({int(current_progress)}% coverage)[/bold green]")
                            break
                            
                        # Get the next question topic
                        if feedback.missing_questions:
                            # Pick ONE missing topic for the next question
                            valid_indices = [i for i in feedback.missing_questions if 0 <= i < len(questions)]
                            if valid_indices:
                                next_topic_idx = valid_indices[0]
                                next_topic = questions[next_topic_idx].text
                            else:
                                # No valid missing questions, pick the next in sequence
                                next_topic_idx = turn % len(questions)
                                next_topic = questions[next_topic_idx].text
                        else:
                            # No missing questions but still below threshold
                            if current_progress < self.completion_threshold:
                                console.print(f"⚠️ [bold yellow]Progress is only {int(current_progress)}%. Let's dive deeper.[/bold yellow]")
                            
                            # Pick the next question in sequence
                            next_topic_idx = turn % len(questions)
                            next_topic = questions[next_topic_idx].text
                    except Exception as e:
                        console.print(f"[bold red]Error parsing evaluation: {str(e)}[/bold red]")
                        # Pick the next question in sequence
                        next_topic_idx = turn % len(questions)
                        next_topic = questions[next_topic_idx].text
            except Exception as e:
                console.print(f"[bold red]Error evaluating responses: {str(e)}[/bold red]")
                # Pick the next question in sequence
                next_topic_idx = turn % len(questions)
                next_topic = questions[next_topic_idx].text
            
            # Generate next conversational question
            follow_text = (
                f"Based on the conversation so far:\n{interview_content}\n\n"
                f"Ask ONE simple follow-up question about: {next_topic}\n"
                f"Make it conversational and easy to answer. Don't use bullet points or numbering."
            )
            
            # Show that the agent is working
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Preparing follow-up question...[/bold blue]"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("Preparing", total=None)
                success, followup = self.run_agent("interviewer", follow_text)
                progress.update(task, completed=True)
                
            if not success:
                console.print("[bold red]Failed to get follow-up question. Using generic question.[/bold red]")
                followup = f"Can you tell me more about {next_topic}?"
                
            console.print("\n" + followup)
            
            # Get response and continue conversation
            next_response = self._get_user_input("\n[bold magenta]>> Your response[/bold magenta]")
            
            # Check for exit commands
            if next_response.lower() == "exit":
                # Exit early
                console.print("\n[bold yellow]Exiting section at user request[/bold yellow]")
                return False, interview_content, []
            
            interview_content += "\n" + followup + "\n" + next_response
            
            # Check if this is the last turn
            if turn == self.max_turns_per_section - 1:
                console.print(f"⚠️ [bold yellow]Maximum conversation turns reached. Final progress: {int(current_progress)}%[/bold yellow]")
        
        # Synthesize answers for this section
        console.print(f"\n🔄 [bold blue]Synthesizing answers for section: {title}...[/bold blue]")
        section_answers = []
        
        for question in questions:
            synth_prompt = (
                f"QUESTION: {question.text}\n\n"
                f"CONVERSATION TRANSCRIPT:\n{interview_content}\n\n"
                f"Based on this conversation, synthesize a comprehensive, coherent answer to the question. "
                f"Your response MUST be ONLY a JSON object with keys \"question\" and \"answer\"."
            )
            
            try:
                # Show that the agent is working
                with Progress(
                    SpinnerColumn(),
                    TextColumn(f"[bold blue]Synthesizing answer for question {question.index+1}...[/bold blue]"),
                    console=console,
                    transient=True,
                ) as progress:
                    task = progress.add_task("Synthesizing", total=None)
                    success, synth_resp_raw = self.run_agent("synthesizer", synth_prompt)
                    progress.update(task, completed=True)
                
                synthesized_answer_text = "Unable to synthesize an answer from the conversation." # Default

                if success and synth_resp_raw:
                    # Attempt to strip markdown code fences from the raw response
                    cleaned_synth_resp = synth_resp_raw.strip()
                    if cleaned_synth_resp.startswith("```json") and cleaned_synth_resp.endswith("```"):
                        # Strip ```json ... ```
                        cleaned_synth_resp = cleaned_synth_resp[len("```json"):].rstrip("`").strip()
                    elif cleaned_synth_resp.startswith("```") and cleaned_synth_resp.endswith("```"):
                        # Strip generic ``` ... ```
                        cleaned_synth_resp = cleaned_synth_resp[len("```"):].rstrip("`").strip()
                    
                    try:
                        # Try to parse the cleaned response as JSON
                        response_data = json.loads(cleaned_synth_resp)
                        # Extract the 'answer' field from the JSON object
                        if isinstance(response_data, dict) and 'answer' in response_data:
                            synthesized_answer_text = response_data['answer']
                        elif isinstance(response_data, str): # Agent returned a JSON encoded string e.g. "\"My answer\""
                            synthesized_answer_text = response_data
                        else:
                            # Parsed to JSON, but not the expected dict structure or a simple string.
                            synthesized_answer_text = cleaned_synth_resp # Use the cleaned (unfenced) string
                            console.print(f"[yellow]Warning: Synthesizer returned parsable but unexpected JSON structure: {cleaned_synth_resp[:100]}...[/yellow]")

                    except json.JSONDecodeError:
                        # If JSON parsing fails even after cleaning, use the cleaned string directly.
                        synthesized_answer_text = cleaned_synth_resp 
                        console.print(f"[yellow]Warning: Synthesizer response was not valid JSON after cleaning: {cleaned_synth_resp[:100]}...[/yellow]")
                    except TypeError: 
                        console.print(f"[yellow]Warning: Synthesizer response was not processable as string for JSON parsing.[/yellow]")
                        # synthesized_answer_text remains the default "Unable to synthesize..."
                
                section_answers.append({
                    "question": question.text,
                    "answer": synthesized_answer_text # This should now be the plain text answer
                })
                    
            except Exception as e:
                console.print(f"[bold red]Error synthesizing answer for q {question.index+1}: {str(e)}[/bold red]")
                section_answers.append({
                    "question": question.text,
                    "answer": "Error during synthesis process."
                })
        
        return True, interview_content, section_answers
        
    def _parse_feedback(self, feedback_text: str) -> SectionFeedback:
        """
        Parse the feedback from the evaluator agent.
        
        Args:
            feedback_text: The text response from the evaluator
            
        Returns:
            SectionFeedback: The parsed feedback
        """
        try:
            # Try to parse as JSON
            data = json.loads(feedback_text)
            return SectionFeedback(
                content=data.get("content", ""),
                is_complete=data.get("is_complete", False),
                missing_questions=data.get("missing_questions", []),
                completion_percentage=data.get("completion_percentage", 0.0)
            )
        except json.JSONDecodeError:
            # If not valid JSON, try to extract information from the text
            console.print("[yellow]Warning: Could not parse feedback as JSON. Extracting information from text.[/yellow]")
            
            # Default values
            content = feedback_text
            is_complete = "complete" in feedback_text.lower() and "not complete" not in feedback_text.lower()
            
            # Try to extract missing questions
            missing_questions = []
            missing_match = re.search(r"missing_questions\s*[=:]\s*\[(.*?)\]", feedback_text, re.IGNORECASE)
            if missing_match:
                try:
                    missing_str = missing_match.group(1).strip()
                    if missing_str:
                        missing_questions = [int(x.strip()) for x in missing_str.split(",")]
                except:
                    pass
            
            # Try to extract completion percentage
            completion_percentage = 0.0 # Default to float
            percentage_match = re.search(r"completion_percentage\s*[=:]\s*(\d+\.?\d*|\d+)", feedback_text, re.IGNORECASE)
            if percentage_match:
                try:
                    completion_percentage = float(percentage_match.group(1))
                except:
                    pass # Keep default if conversion fails
            
            return SectionFeedback(
                content=content,
                is_complete=is_complete,
                missing_questions=missing_questions,
                completion_percentage=completion_percentage
            )

    def _save_results(self, result: QuestionnaireResult) -> str:
        """
        Save the results to a file.
        
        Args:
            result: The questionnaire result
            
        Returns:
            str: The path to the saved file
        """
        # Generate timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"qa_result_{timestamp}.md"
        
        # Format synthesized answers
        final_output = []
        
        # Add preamble if it exists
        if self._preamble_content:
            final_output.append(self._preamble_content)

        for section_title, answers in result.synthesized_answers.items():
            section_output = [f"## {section_title}"]
            
            for idx, qa in enumerate(answers):
                section_output.append(f"**{idx + 1}. Q: {qa['question']}**") # Added enumeration
                # The 'answer' should now be plain text due to processing in _process_section
                answer_text = qa.get('answer', "Response not available.")
                if not isinstance(answer_text, str):
                    answer_text = str(answer_text) # Ensure it's a string
                section_output.append(answer_text) # Append the plain text answer
                section_output.append("")  # Empty line for spacing
            
            final_output.append("\n".join(section_output))
        
        synthesized = "\n\n".join(final_output)
        
        # Write to file
        with open(output_file, "w") as f:
            f.write(synthesized)
            
        console.print(f"[bold green]Saved results to: {output_file}[/bold green]")
        
        return output_file

    def cleanup(self):
        """Delete the agents when done."""
        if self.client and self.agents:
            console.print("[dim]Cleaning up agents...[/dim]")
            for agent_name, agent in self.agents.items():
                try:
                    self.client.agents.delete_agent(agent.id)
                    console.print(f"[dim]Deleted {agent_name} agent[/dim]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not delete {agent_name} agent: {str(e)}[/yellow]")

    def _update_temp_results(self, all_responses: Dict[str, str], synthesized_answers: Dict[str, List[Dict[str, str]]], current_section: str = None):
        """
        Update the temporary results file with the current progress.
        
        Args:
            all_responses: Dictionary of section titles to conversation contents
            synthesized_answers: Dictionary of section titles to lists of Q&A pairs
            current_section: Optional name of the section currently being processed
        """
        try:
            # Format synthesized answers for the temporary file
            temp_output = []
            
            # Add a header with timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            temp_output.append(f"# Temporary Results (Last Updated: {timestamp})\n")

            # Add preamble to temp results as well
            if self._preamble_content:
                temp_output.append(self._preamble_content)

            temp_output.append("## Progress Summary")
            
            # Use cached sections instead of parsing again
            total_sections = len(self._parsed_sections) if self._parsed_sections else 0
            completed_sections = len(synthesized_answers)
            
            temp_output.append(f"Completed sections: {completed_sections} of {total_sections}\n")
            
            # Show current section being processed if provided
            if current_section:
                temp_output.append(f"**Currently processing:** {current_section}\n")
            
            # Add the results of processed sections
            for section_title, answers in synthesized_answers.items():
                section_output = [f"## {section_title}"]
                
                for idx, qa in enumerate(answers):
                    section_output.append(f"**{idx + 1}. Q: {qa['question']}**") # Added enumeration
                    section_output.append(f"{qa['answer']}")
                    section_output.append("")  # Empty line for spacing
                
                temp_output.append("\n".join(section_output))
            
            # Write to temporary file
            with open(self.temp_results_file, "w") as f:
                f.write("\n\n".join(temp_output))
                
            console.print(f"[green]Updated temporary results in: {self.temp_results_file}[/green]")
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not update temporary results file: {str(e)}[/yellow]")
    
    def _delete_temp_results(self):
        """
        Delete the temporary results file if it exists.
        """
        try:
            if os.path.exists(self.temp_results_file):
                os.remove(self.temp_results_file)
                console.print(f"[dim]Deleted temporary results file: {self.temp_results_file}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not delete temporary results file: {str(e)}[/yellow]")

def main():
    """Main function for running the questionnaire."""
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process a questionnaire using Azure AI.')
    parser.add_argument('questions_file', help='Path to the markdown file containing questions')
    parser.add_argument('--unfinished', '-u', help='Path to unfinished Q&A file (optional)')
    parser.add_argument('--max-turns', '-t', type=int, default=5,
                      help='Maximum number of conversation turns per section (default: 5)')
    parser.add_argument('--completion-threshold', '-c', type=int, default=90,
                      help='Completion threshold percentage (0-100) to consider a section complete (default: 90)')
    
    args = parser.parse_args()
    
    # Check connection string
    conn_str = os.getenv("AZURE_AI_CONN_STR", AZURE_AI_CONN_STR)
    if not check_connection_string(conn_str):
        console.print("[bold red]Failed to connect to Azure AI. Please check your connection string and try again.[/bold red]")
        sys.exit(1)

    # Display available models
    console.print("\nSelect model:")
    for i, (model_id, model_info) in enumerate(AZURE_MODELS.items(), 1):
        description = model_info["description"]
        console.print(f"{i}: {model_id} - {description}")
    
    # Get user choice with default
    model_choices = {str(i): model_id for i, model_id in enumerate(AZURE_MODELS.keys(), 1)}
    model_choice = Prompt.ask("\n[bold magenta]Enter your choice[/bold magenta]", choices=list(model_choices.keys()), default="1")
    selected_model = list(AZURE_MODELS.keys())[int(model_choice)-1]
    
    console.print(f"[green]Selected model: {selected_model}[/green]")
    console.print(f"[green]Using questions from: {args.questions_file}[/green]")
    console.print(f"[green]Max turns per section: {args.max_turns}[/green]")
    console.print(f"[green]Completion threshold: {args.completion_threshold}%[/green]")
    
    try:
        # Read questions file with proper encoding and character replacement
        try:
            with open(args.questions_file, 'r', encoding='utf-8') as f:
                md = f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 if utf-8 fails
            with open(args.questions_file, 'r', encoding='latin-1') as f:
                md = f.read()
        
        # Replace problematic characters
        md = md.replace('–', '-')  # Replace en dash with regular hyphen
        md = md.replace('—', '-')  # Replace em dash with regular hyphen
        
        # Handle unfinished Q&A file
        unfinished_content = None
        unfinished_file = args.unfinished
        
        # If no unfinished file provided via command line, prompt the user
        if not unfinished_file:
            use_unfinished = Prompt.ask(
                "[bold magenta]Do you want to use a file with unfinished Q&A content?[/bold magenta]", 
                choices=["y", "n"], 
                default="n"
            )
            if use_unfinished.lower() == "y":
                unfinished_file = Prompt.ask("Enter the path to your unfinished Q&A file")
        
        # Read unfinished Q&A file if provided (either via command line or prompt)
        if unfinished_file:
            try:
                try:
                    with open(unfinished_file, 'r', encoding='utf-8') as f:
                        unfinished_content = f.read()
                except UnicodeDecodeError:
                    # Fallback to latin-1 if utf-8 fails
                    with open(unfinished_file, 'r', encoding='latin-1') as f:
                        unfinished_content = f.read()
                
                # Replace problematic characters in unfinished content
                unfinished_content = unfinished_content.replace('–', '-')
                unfinished_content = unfinished_content.replace('—', '-')
                
                console.print(f"[green]Loaded unfinished Q&A from: {unfinished_file}[/green]")
            except FileNotFoundError:
                console.print(f"[yellow]Warning: Unfinished Q&A file '{unfinished_file}' not found. Starting from scratch.[/yellow]")
        
        # Initialize and run the questionnaire
        questionnaire = AzureAIQuestionnaire(model_id=selected_model, conn_str=conn_str)
        
        if not questionnaire.setup():
            console.print("[bold red]Failed to set up questionnaire. Exiting.[/bold red]")
            sys.exit(1)
        
        # Set the custom parameters
        questionnaire.max_turns_per_section = args.max_turns
        questionnaire.completion_threshold = args.completion_threshold
            
        result = questionnaire.set_input(md, unfinished_qa=unfinished_content).process_questionnaire()
        
        # Cleanup happens within the process_questionnaire method
        questionnaire.cleanup()
        
        console.print("\n[bold green]Questionnaire complete![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    main() 