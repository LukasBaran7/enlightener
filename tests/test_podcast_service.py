import pytest
from datetime import datetime, timezone
from app.routers.podcast import PodcastService, Episode, Podcast
from app.repositories.podcast_repository import PodcastRepository
from unittest.mock import Mock, AsyncMock

@pytest.fixture
def mock_repository():
    return Mock(spec=PodcastRepository)

@pytest.fixture
def podcast_service(mock_repository):
    return PodcastService(mock_repository)

@pytest.fixture
def sample_podcast_data():
    return [
        {
            "podcast_title": "Test Podcast",
            "artwork_url": "http://example.com/art.jpg",
            "created_at": "2024-03-20T10:00:00Z",
            "source": "overcast",
            "episodes": [
                {
                    "episode_title": "Episode 1",
                    "audio_url": "http://example.com/ep1.mp3",
                    "overcast_url": "http://overcast.fm/ep1",
                    "overcast_id": "ep1",
                    "published_date": "2024-03-19T10:00:00Z",
                    "play_progress": 100,
                    "last_played_at": "2024-03-20T09:00:00Z",
                    "summary": "Test summary"
                },
                {
                    "episode_title": "Episode 2",
                    "audio_url": "http://example.com/ep2.mp3",
                    "overcast_url": "http://overcast.fm/ep2",
                    "overcast_id": "ep2",
                    "published_date": "2024-03-18T10:00:00Z",
                    "play_progress": 50,
                    "last_played_at": "2024-03-19T09:00:00Z",
                    "summary": "Test summary 2"
                }
            ]
        }
    ]

@pytest.mark.asyncio
async def test_get_latest_episodes_happy_path(podcast_service, mock_repository, sample_podcast_data):
    # Arrange
    mock_repository.find_all = AsyncMock(return_value=sample_podcast_data)

    # Act
    result = await podcast_service.get_latest_episodes()

    # Assert
    assert len(result) == 1
    assert result[0].podcast_title == "Test Podcast"
    assert len(result[0].episodes) == 2
    assert result[0].episodes[0].episode_title == "Episode 1"
    assert result[0].episodes[1].episode_title == "Episode 2"

def test_process_date_with_valid_string(podcast_service):
    # Arrange
    test_date = "2024-03-20T10:00:00Z"

    # Act
    result = podcast_service.process_date(test_date)

    # Assert
    assert result == "2024-03-20T10:00:00+00:00"

def test_process_date_with_datetime(podcast_service):
    # Arrange
    test_date = datetime(2024, 3, 20, 10, 0, 0, tzinfo=timezone.utc)

    # Act
    result = podcast_service.process_date(test_date)

    # Assert
    assert result == "2024-03-20T10:00:00+00:00"

def test_create_episode_from_doc(podcast_service):
    # Arrange
    episode_doc = {
        "episode_title": "Test Episode",
        "audio_url": "http://example.com/test.mp3",
        "overcast_url": "http://overcast.fm/test",
        "overcast_id": "test123",
        "published_date": "2024-03-20T10:00:00Z",
        "play_progress": 50,
        "last_played_at": "2024-03-20T11:00:00Z",
        "summary": "Test summary"
    }

    # Act
    episode = podcast_service._create_episode(episode_doc)

    # Assert
    assert isinstance(episode, Episode)
    assert episode.episode_title == "Test Episode"
    assert episode.audio_url == "http://example.com/test.mp3"
    assert episode.play_progress == 50 