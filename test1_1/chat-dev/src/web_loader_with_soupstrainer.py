from langchain_community.document_loaders import WebBaseLoader
from sentence_transformers import SentenceTransformer
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from bs4 import SoupStrainer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_web_loader_with_soupstrainer(urls, parse_only=None):
    """
    SoupStrainer를 사용하여 특정 태그만 파싱하는 웹 로더를 생성합니다.
    
    Args:
        urls: URL 또는 URL 리스트
        parse_only: SoupStrainer 객체 (예: SoupStrainer("p") for p 태그만)
    
    Returns:
        WebBaseLoader: 설정된 웹 로더
    """
    if parse_only is None:
        parse_only = SoupStrainer("p")  # 기본값으로 p 태그만 파싱
    
    loader = WebBaseLoader(
        urls,
        bs_kwargs={
            "parse_only": parse_only,  # 텍스트 노드만 파싱
        }
    )
    return loader

def embed_and_upload_from_web_with_filtering(collection_name, urls, parse_only=None):
    """
    웹에서 문서를 로드하고, 특정 태그만 파싱하여 Qdrant에 업로드합니다.
    
    Args:
        collection_name: Qdrant 컬렉션 이름
        urls: URL 리스트
        parse_only: SoupStrainer 객체 (예: SoupStrainer("p") for p 태그만)
    """
    # 임베딩 모델 로드
    embedding_model = SentenceTransformer('./models/BGE-m3-ko')
    
    # Qdrant 클라이언트 설정
    qdrant_client = QdrantClient(
        host="localhost",
        port=6333
    )
    
    # 웹에서 문서 로드 (SoupStrainer 사용)
    loader = create_web_loader_with_soupstrainer(urls, parse_only)
    docs = loader.load()
    texts = [doc.page_content for doc in docs]
    
    if not texts:
        logger.warning("로드된 텍스트가 없습니다.")
        return
    
    # 텍스트 임베딩 생성
    embeddings = embedding_model.encode(texts).tolist()
    
    # Qdrant에 업로드할 포인트 데이터 생성
    points = [
        qdrant_client.models.PointStruct(
            id=i,
            vector=embeddings[i],
            payload={"text": texts[i]}
        )
        for i in range(len(texts))
    ]
    
    # 컬렉션이 없으면 생성
    if collection_name not in [c.name for c in qdrant_client.get_collections().collections]:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=qdrant_client.models.VectorParams(
                size=len(embeddings[0]),
                distance="Cosine"
            )
        )
    
    # 포인트 업로드
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )
    logger.info(f"{len(points)}개의 웹 문서가 Qdrant에 임베딩되어 업로드되었습니다.")

# 사용 예시
if __name__ == "__main__":
    urls = [
        "https://www.pcninc.co.kr/",
        "https://www.pcninc.co.kr/digital/ai.do",
        "https://www.pcninc.co.kr/digital/bigdata.do",
        "https://www.pcninc.co.kr/digital/xrcontents.do",
        "https://www.pcninc.co.kr/digital/portfolio/list.do",
        "https://www.pcninc.co.kr/siux/public.do",
        "https://www.pcninc.co.kr/siux/finance.do",
        "https://www.pcninc.co.kr/siux/brand.do",
        "https://www.pcninc.co.kr/siux/health.do",
        "https://www.pcninc.co.kr/solution/oasis.do",
        "https://www.pcninc.co.kr/solution/apim.do",
        "https://www.pcninc.co.kr/solution/esearch.do",
        "https://www.pcninc.co.kr/solution/oasisx.do",
        "https://www.pcninc.co.kr/solution/datamap.do",
        "https://www.pcninc.co.kr/solution/trenddata.do",
        "https://www.pcninc.co.kr/solution/ozai.do",
        "https://www.pcninc.co.kr/company/introduce.do",
        "https://www.pcninc.co.kr/company/business.do?accYear=2023",
        "https://www.pcninc.co.kr/company/benefit.do",
        "https://www.pcninc.co.kr/company/history.do",
        "https://www.pcninc.co.kr/company/location.do",
        "https://www.pcninc.co.kr/ir/disinfo/list.do?page=1&pageSize=10",
        "https://www.pcninc.co.kr/notice/press/list.do?page=1&pageSize=6",
        "https://www.pcninc.co.kr/notice/plus/list.do?page=1&pageSize=6",
        "https://www.pcninc.co.kr/notice/news/list.do?page=1&pageSize=6",
    ]
    
    # p 태그만 파싱하여 업로드
    embed_and_upload_from_web_with_filtering("pcn_web_intro_p_only", urls, SoupStrainer("p"))
    
    # 다른 태그들도 파싱하고 싶다면:
    # embed_and_upload_from_web_with_filtering("pcn_web_intro_multiple", urls, 
    #     SoupStrainer(["p", "h1", "h2", "h3", "h4", "h5", "h6"])) 