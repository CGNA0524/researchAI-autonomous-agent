"""
Tool Registry for dynamic tool discovery, schema provisioning, and dispatch.
"""

from typing import Any, Dict, List, Optional
from app.tools.base import BaseTool
from app.tools.web_search import WebSearchTool
from app.tools.webpage_extractor import WebpageExtractorTool
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """Manages available agent research tools and their invocations."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}
        # Register standard default research tools
        self.register(WebSearchTool())
        self.register(WebpageExtractorTool())

    def register(self, tool: BaseTool) -> None:
        """Registers a tool instance into the registry."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Retrieves a tool by name."""
        return self._tools.get(name)

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Returns the list of tool schemas formatted for LLM tool-calling."""
        return [tool.get_schema() for tool in self._tools.values()]

    async def execute(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        """Safely dispatches execution to the registered tool."""
        tool = self.get_tool(tool_name)
        if not tool:
            logger.error(f"Attempted to execute unregistered tool: {tool_name}")
            return {
                "success": False,
                "error": f"Tool '{tool_name}' is not registered in the system."
            }
        return await tool.execute(**kwargs)
