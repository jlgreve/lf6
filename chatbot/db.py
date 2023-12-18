from typing import Optional, List

from sqlalchemy import String, ForeignKey, create_engine, Engine
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped, relationship


class Base(DeclarativeBase):
    pass


class ChatHistory(Base):
    __tablename__ = 'chat_history'

    id: Mapped[int] = mapped_column(primary_key=True)
    time_start: Mapped[int] = mapped_column()
    time_end: Mapped[Optional[int]] = mapped_column()

    messages: Mapped[List['ChatMessage']] = relationship(
        back_populates='chat', cascade='all, delete-orphan'
    )

    feedback: Mapped['ChatFeedback'] = relationship(
        back_populates='chat'
    )

    def __repr__(self):
        return f'ChatHistory(id={self.id!r}, time_start={self.time_start!r}, time_end={self.time_end!r})'


class ChatMessage(Base):
    __tablename__ = 'chat_message'

    id: Mapped[int] = mapped_column(primary_key=True)
    chat_id: Mapped[int] = mapped_column(ForeignKey('chat_history.id'))
    from_support: Mapped[bool] = mapped_column()
    time_sent: Mapped[int] = mapped_column()
    content: Mapped[str] = mapped_column(String(1024))

    chat: Mapped['ChatMessage'] = relationship(back_populates='messages')

    def __repr__(self):
        return f'ChatHistory(id={self.id!r}, chat_id={self.chat_id!r}, from_support={self.from_support!r}, time_sent={self.time_sent!r}, content={self.content!r})'


class ChatFeedback(Base):
    __tablename__ = 'chat_feedback'

    id: Mapped[int] = mapped_column(ForeignKey('chat_history.id'), primary_key=True)
    stars: Mapped[int] = mapped_column()

    chat: Mapped['chat'] = relationship(back_populates='feedback')

    def __repr__(self):
        return f'ChatFeedback(id={self.id!r})'


def sql_default_engine() -> Engine:
    return create_engine('sqlite://')
