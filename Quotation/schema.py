from typing import Optional, List
from pydantic import BaseModel, Field


class Risk(BaseModel):
    """
    Represents a potential issue or uncertainty that might impact the project.
    Each risk includes:
      - A concise name to identify it.
      - A detailed description outlining its nature.
      - An optional risk level indicating severity (e.g., 'Low', 'Medium', 'High').
    """

    riskName: str = Field(
        ...,
        description="A short label identifying the risk (e.g., 'Security Vulnerability')."
    )
    riskDescription: str = Field(
        ...,
        description="A comprehensive explanation of the risk, including possible causes and impacts."
    )
    riskLevel: Optional[str] = Field(
        None,
        description="An optional indicator of the risk's severity level (e.g., 'Low', 'Medium', 'High')."
    )


class Task(BaseModel):
    """
    Represents a discrete work item or action within a phase.
    Each task includes:
      - A name that succinctly describes its purpose.
      - A detailed explanation of what the task entails.
      - An estimated effort in person-days.
      - A list of risk elements specific to this task.
    """

    taskName: str = Field(
        ...,
        description="Name or title of the task (e.g., 'Requirements Gathering')."
    )
    taskDescription: str = Field(
        ...,
        description="A detailed explanation of the task's objectives and scope."
    )
    estimatedEffort: float = Field(
        ...,
        description="Approximate effort required to complete this task, in person-days."
    )
    risks: List[Risk] = Field(
        default_factory=list,
        description="A list of risk factors specifically related to this task."
    )


class Phase(BaseModel):
    """
    Represents a major segment or milestone within the project.
    A phase can include:
      - A name to identify the phase.
      - A brief description of its focus.
      - A set of tasks that need to be completed during this phase.
    """

    phaseName: str = Field(
        ...,
        description="The name of the phase (e.g., 'Requirements Definition', 'Design', 'Development', 'Testing')."
    )
    phaseDescription: Optional[str] = Field(
        None,
        description="A textual overview of what this phase covers or aims to achieve."
    )
    tasks: List[Task] = Field(
        default_factory=list,
        description="A collection of tasks to be carried out within this phase."
    )


class Project(BaseModel):
    """
    Encapsulates the entire project quotation, including:
      - The project's name.
      - All phases required for completion.
      - Project-wide risks.
      - A detailed description of the project scope.
    """

    projectName: str = Field(
        ...,
        description="The official or working name of the project."
    )
    phases: List[Phase] = Field(
        default_factory=list,
        description="An ordered list of phases the project will go through."
    )
    overallRisks: List[Risk] = Field(
        default_factory=list,
        description="Risks that have broad impact across multiple phases or the entire project."
    )
    description: str = Field(
        ...,
        description="A detailed explanation of the project's goals, deliverables, and scope."
    )


def get_project_quotation_tool() -> dict:
    """
    Provides a specification for a tool that generates project quotations.
    This tool:
      - Extracts information from input documents.
      - Maps the extracted data to a 'Project' schema.
      - Yields a structured output representing a software or design production cost quotation.
    """
    return {
        "toolSpec": {
            "name": "project_quotation",
            "description": (
                "This tool is used for extracting or parsing information from documents "
                "and mapping it to a Project schema, representing a software or design production cost quotation."
            ),
            "inputSchema": {"json": Project.model_json_schema()},
        }
    }