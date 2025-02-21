from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict


class FocusGroup(BaseModel):
    focus_group: Optional[int] = Field(description="The focus group number")
    date: Optional[str] = Field(description="The date of the focus group")
    participants: Optional[List[str]] = Field(description="The participants of the focus group")
    content: str = Field(description="The content of the focus group excluding lines with focus group number and date")


class CodeExcerpt(BaseModel):
    code: str = Field(description="The code or theme")
    code_description: str = Field(description="The description of the code")
    excerpt: str = Field(description="The excerpt supporting the code")
    speaker: Optional[str] = Field(description="The speaker of the line")


class Themes(BaseModel):
    theme: str = Field(description="The themes of the text")
    theme_definition: str = Field(description="The definition of the themes")
    subthemes: List[str] = Field(description="The subthemes of the theme")
    subtheme_definitions: List[str] = Field(description="The definitions of the subthemes")
    supporting_quotes: List[str] = Field(description="The supporting quotes for the theme or subthemes")


class ZSControl(BaseModel):
    theme: str = Field(description="The theme of the text")
    theme_definition: str = Field(description="The definition of the theme")
    subthemes: List[str] = Field(description="The subthemes of the theme")
    subtheme_definitions: List[str] = Field(description="The definitions of the subthemes")
    codes: List[str] = Field(description="The supporting codes for the theme or subthemes")
    supporting_quotes: str = Field(description="The excerpt supporting the code")
    speaker: str = Field(description="The speaker of the line denoted by TN")