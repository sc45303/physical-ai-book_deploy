---

description: "Task list for Book-based RAG System implementation"
---

# Tasks: Book-based RAG System with FastAPI, PostgreSQL and Qdrant Vector Database

**Input**: Design documents from `/specs/002-book-rag-system/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Web app**: `backend/src/`, `frontend/src/`
- **Backend directory structure based on plan.md**: `backend/`
- Paths adjusted for the specific project structure

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create backend project structure in backend/
- [X] T002 Initialize Python project with FastAPI, SQLAlchemy, Pydantic dependencies in backend/requirements.txt
- [X] T003 Create .env.example file with required environment variables per quickstart.md
- [X] T004 [P] Configure linting and formatting tools (Black, Flake8, mypy) in backend/
- [X] T005 Set up database migration framework (Alembic) in backend/

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T006 Setup database connection and initialization in backend/app/database/database.py
- [X] T007 [P] Create base models with common fields in backend/app/models/base.py
- [X] T008 [P] Setup configuration management for API keys in backend/app/config.py
- [X] T009 Create API response models in backend/app/models/responses.py
- [X] T010 [P] Setup error handling and custom exceptions in backend/app/exceptions.py
- [X] T011 Setup logging infrastructure in backend/app/utils/logging.py
- [X] T012 [P] Setup Qdrant vector store connection in backend/app/database/vector_store.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Upload and Process Book Content (Priority: P1) 🎯 MVP

**Goal**: Enable users to upload books in various formats (PDF, DOCX, TXT) and process them into searchable chunks

**Independent Test**: Can be fully tested by uploading a book file and verifying that it's properly stored and processed into searchable chunks

### Implementation for User Story 1

- [X] T013 [P] [US1] Create User model in backend/app/models/user.py
- [X] T014 [P] [US1] Create Book model in backend/app/models/book.py
- [X] T015 [P] [US1] Create BookChunk model in backend/app/models/book_chunk.py
- [X] T016 [P] [US1] Create VectorEmbedding model in backend/app/models/vector_embedding.py
- [X] T017 [US1] Implement document parser service in backend/app/services/document_parser.py
- [X] T018 [US1] Implement text chunking service in backend/app/services/chunking_service.py
- [X] T019 [US1] Implement file upload handler in backend/app/services/file_service.py
- [X] T020 [US1] Implement embedding service in backend/app/services/embedding_service.py
- [X] T021 [US1] Create upload endpoint in backend/app/routes/upload.py
- [X] T022 [US1] Create endpoint for checking processing status in backend/app/routes/status.py
- [X] T023 [US1] Integrate upload functionality with Qdrant vector storage
- [X] T024 [US1] Add validation for file formats and size limits
- [X] T025 [US1] Add database migrations for new models
- [X] T026 [US1] Add logging for book processing operations

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Ask Questions About Book Content (Priority: P2)

**Goal**: Allow users to ask questions about book content and receive accurate answers based on book's information

**Independent Test**: Can be fully tested by asking questions about the book content and receiving relevant answers that reference or extract information from the book

### Implementation for User Story 2

- [X] T027 [P] [US2] Create ChatSession model in backend/app/models/chat_session.py
- [X] T028 [P] [US2] Create ChatMessage model in backend/app/models/chat_message.py
- [X] T029 [US2] Implement RAG service for question answering in backend/app/services/rag_service.py
- [X] T030 [US2] Implement LLM service integration in backend/app/services/llm_service.py
- [X] T031 [US2] Implement semantic search functionality in backend/app/services/search_service.py
- [X] T032 [US2] Create chat endpoint for global mode in backend/app/routes/chat.py
- [X] T033 [US2] Implement context window management and source citations
- [X] T034 [US2] Add conversation history storage and retrieval
- [X] T035 [US2] Ensure answers are grounded in book content without hallucination

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Search Through Book Content (Priority: P3)

**Goal**: Enable users to search for specific terms, concepts, or topics within the book

**Independent Test**: Can be fully tested by searching for specific terms in the book and receiving relevant sections or paragraphs

### Implementation for User Story 3

- [X] T036 [P] [US3] Enhance search service to support keyword search in backend/app/services/search_service.py
- [X] T037 [US3] Create dedicated search endpoint in backend/app/routes/search.py
- [X] T038 [US3] Implement full-text search capabilities integrating with PostgreSQL
- [X] T039 [US3] Add search result ranking and relevance scoring
- [X] T040 [US3] Implement caching for frequent search queries

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: User Story 4 - Answer Questions from Selected Text (Priority: P3)

**Goal**: Allow users to select specific text and ask questions only about that selected text

**Independent Test**: Can be fully tested by selecting text in the book and asking questions that are answered only from the selected text

### Implementation for User Story 4

- [X] T041 [US4] Create endpoint for selected text chat in backend/app/routes/chat.py
- [X] T042 [US4] Modify RAG service to support selected text mode in backend/app/services/rag_service.py
- [X] T043 [US4] Implement text selection preprocessing and validation
- [ ] T044 [US4] Ensure answers are strictly based on selected text only
- [ ] T045 [US4] Add UI indicators for selected text mode to response

**Checkpoint**: All user stories should now be independently functional

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T046 [P] Documentation updates for all API endpoints in backend/docs/
- [ ] T047 Code cleanup and refactoring across all modules
- [ ] T048 Performance optimization for vector search and LLM calls
- [ ] T049 [P] Add comprehensive unit tests for core services in backend/tests/
- [ ] T050 Security hardening for file uploads and API access
- [ ] T051 [P] Add API rate limiting and monitoring
- [ ] T052 Run quickstart.md validation
- [ ] T053 Add health check endpoints
- [ ] T054 Optimize database queries and add indexing

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Depends on US1 (Book and BookChunk models)
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Depends on US1 (Book models)
- **User Story 4 (P3)**: Can start after Foundational (Phase 2) - Depends on US1 and US2

### Within Each User Story

- Models need to be created first
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all models for User Story 1 together:
Task: "Create User model in backend/app/models/user.py"
Task: "Create Book model in backend/app/models/book.py"
Task: "Create BookChunk model in backend/app/models/book_chunk.py"
Task: "Create VectorEmbedding model in backend/app/models/vector_embedding.py"

# Launch all services for User Story 1 together (after models):
Task: "Implement document parser service in backend/app/services/document_parser.py"
Task: "Implement text chunking service in backend/app/services/chunking_service.py"
Task: "Implement embedding service in backend/app/services/embedding_service.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Add User Story 4 → Test independently → Deploy/Demo
6. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2 (after US1 models created)
   - Developer C: User Story 3 (after US1 models created)
   - Developer D: User Story 4 (after US1/US2 completed)
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Focus on User Story 1 first as the MVP for the RAG system