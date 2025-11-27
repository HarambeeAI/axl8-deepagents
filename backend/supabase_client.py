"""Supabase client for the Deep Agents backend."""

import os
from functools import lru_cache

from supabase import create_client, Client


@lru_cache()
def get_supabase_client() -> Client:
    """Get a cached Supabase client instance.
    
    Returns:
        Supabase client configured with environment variables.
    
    Raises:
        ValueError: If required environment variables are not set.
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    
    if not url:
        raise ValueError("SUPABASE_URL environment variable is required")
    if not key:
        raise ValueError("SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY environment variable is required")
    
    return create_client(url, key)


def get_database_url() -> str:
    """Get the database URL for LangGraph checkpointing.
    
    Returns:
        PostgreSQL connection string for Supabase.
    
    Raises:
        ValueError: If DATABASE_URL is not set.
    """
    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL environment variable is required for checkpointing")
    return url
