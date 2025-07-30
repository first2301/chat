import logging
from typing import List, Union
from urllib.parse import urlparse
import time

from langchain_community.document_loaders import WebBaseLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except Exception:
        return False

def load_web_documents(
    urls: Union[str, List[str]],
    timeout: int = 60,
    max_retries: int = 5,
    user_agent: str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
) -> List:
    """
    웹 URL에서 문서를 로드하여 반환합니다.

    Args:
        urls: URL 또는 URL 리스트
        timeout: 요청 타임아웃(초)
        max_retries: 최대 재시도 횟수
        user_agent: User-Agent 헤더

    Returns:
        List: 로드된 문서 객체 리스트
    """
    if isinstance(urls, str):
        urls = [urls]
    elif not isinstance(urls, list):
        raise ValueError("urls는 문자열 또는 문자열 리스트여야 합니다.")

    all_docs = []
    for i, url in enumerate(urls, 1):
        logger.info(f"URL 처리 중 ({i}/{len(urls)}): {url}")
        if not is_valid_url(url):
            logger.warning(f"유효하지 않은 URL: {url}")
            continue

        for attempt in range(max_retries):
            try:
                loader = WebBaseLoader(
                    url,
                    requests_kwargs={
                        'timeout': timeout,
                        'headers': {
                            'User-Agent': user_agent,
                            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                            'Accept-Language': 'ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3',
                            'Accept-Encoding': 'gzip, deflate',
                            'Connection': 'keep-alive',
                        }
                    }
                )
                docs = loader.load()
                logger.info(f"성공적으로 로드됨: {url} ({len(docs)}개 문서)")
                all_docs.extend(docs)
                break
            except Exception as e:
                logger.warning(f"시도 {attempt + 1}/{max_retries} 실패: {url} - {e}")
                if attempt < max_retries - 1:
                    sleep_time = min(2 ** attempt, 30)
                    logger.info(f"{sleep_time}초 대기 후 재시도...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"최종 실패: {url} - {e}")
    return all_docs
