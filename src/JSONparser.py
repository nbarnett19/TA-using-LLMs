from pydantic import BaseModel, Field

class CodeExcerpt(BaseModel):
    code: str = Field(description="The code or theme")
    excerpt: str = Field(description="The excerpt supporting the code")

class ZSThemes(BaseModel):
    theme: str = Field(description="The theme of the text")
    theme_definition: str = Field(description="The definition of the theme")
    subthemes: list[str] = Field(description="The subthemes of the theme")
    subtheme_definitions: list[str] = Field(description="The definitions of the subthemes")
    supporting_quotes: list[str] = Field(description="The supporting quotes for the theme")