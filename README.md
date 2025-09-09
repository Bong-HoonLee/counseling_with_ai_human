# counseling_with_ai_human

## 현재 v2 리뉴얼 작업을 진행 중입니다.

```text
├── app/
│   ├── adapters/
│   │   │── embedding
│   │   │── tokenizers
│   │   └── vectorstore
│   ├── api
│   ├── config/
│   │   └── settings
│   ├── core/
│   │   ├── langgraph
│   │   └── ports
│   ├── infrastructure
│   │   └── vectorstore
│   │       └── qdrant
│   └── models/
│       └── dto
└── scripts/
```

# AI Human Chatbot

> **Qdrant + LangChain + LangGraph 기반 RAG 엔진**과  
> **TTS · Lip Sync · Facial Expression Recognition · Diffusion 모델**을 결합하여  
> 사용자 감정에 반응하는 **AI Human 심리상담 챗봇** 프로젝트입니다.

---

## ✨ 프로젝트 소개

AI Human Chatbot은 단순한 대화형 챗봇을 넘어,  
**감정을 인식하고 사람처럼 반응하는 가상 인간(AI Human)**을 구현하는 것을 목표로 합니다.  

- **AI Human 생성**
  - Diffusion 모델 기반 사실적 얼굴 이미지 생성
  - GAN 기반 Shadow 모델에 Face Swap 적용
  - TTS + GAN 기반 Lip Sync로 음성 및 영상 자동 생성

- **감정 기반 인터랙션**
  - Facial Expression Recognition으로 사용자 표정 인식
  - 감정에 따라 표정과 답변이 달라지는 반응 설계

- **RAG 기반 질의응답 엔진**
  - Qdrant, LangChain, LangGraph 기반 검색·생성 파이프라인
  - 텍스트 전처리 및 벡터화 → 벡터DB 구축
  - 프롬프트 최적화로 LLM 응답 제어

---

## 🧩 프로젝트 배경

마음의 병을 겪는 사람들은 타인과의 **대면 상담에 큰 부담감**을 느끼는 경우가 많습니다.  
이에 따라, 비대면이면서도 **사람처럼 감정에 반응하는 AI Human**이 상담을 대신한다면  
상담의 **감정적 장벽을 낮출 수 있다**고 판단했습니다.  

이러한 필요성에서, **감정을 인식하고 자연스럽게 반응하는 AI 기반 상담 인터페이스**를 기획했습니다.

---

## 🎯 기획 목표

- 감정에 반응하는 **AI Human 챗봇 구현**
- **비대면 감정 소통 환경** 제공
- **몰입감 있는 사용자 경험** 실현

---

## ⚙️ Tech Stack

- **AI Framework**: LangChain, LangGraph  
- **Vector Database**: Qdrant  
- **Multimodal AI**: Diffusion Model, GAN (Face Swap & Lip Sync)  
- **Speech & Vision**: TTS, Facial Expression Recognition  

---

## 🚀 실행 화면

- AI Human 얼굴 생성 샘플
- 감정 인식 및 반응 예시
- 상담 시나리오 시연 GIF/영상



<img width="1317" height="744" alt="image" src="https://github.com/user-attachments/assets/bed28965-9fd2-4264-a143-c82b476e9ace" />
