import time
from typing import Optional, List, Callable, TypeVar

import enum

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import String, ForeignKey, create_engine, Engine, insert, select, Enum
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped, relationship, Session


class Base(DeclarativeBase):
    pass


db = SQLAlchemy(model_class=Base)


class ChatStatusEnum(enum.Enum):
    started = 0
    support_escalated = 1
    pending_feedback = 2
    ended = 3
    pending_resolved = 4


class ChatStatus(db.Model):
    __tablename__ = 'chat_status'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(ForeignKey('chat_history.id'))
    status: Mapped[ChatStatusEnum] = mapped_column(Enum(ChatStatusEnum))
    time_reached: Mapped[int] = mapped_column()
    active: Mapped[bool] = mapped_column()

    history: Mapped['ChatHistory'] = relationship(back_populates='statuses')

    def __repr__(self):
        return f'ChatStatus(id={self.id!r}, chat_id={self.chat_id!r}, status={self.status!r}, time_reached={self.time_reached!r}, active={self.active!r})'


class ChatMessage(db.Model):
    __tablename__ = 'chat_message'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    chat_id: Mapped[int] = mapped_column(ForeignKey('chat_history.id'))
    from_support: Mapped[bool] = mapped_column()
    time_sent: Mapped[int] = mapped_column()
    content: Mapped[str] = mapped_column(String(1024))

    history: Mapped['ChatHistory'] = relationship(back_populates='messages')

    def __repr__(self):
        return f'ChatHistory(id={self.id!r}, chat_id={self.chat_id!r}, from_support={self.from_support!r}, time_sent={self.time_sent!r}, content={self.content!r})'


class ChatResolved(db.Model):
    __tablename__ = 'chat_resolved'

    id: Mapped[int] = mapped_column(ForeignKey('chat_history.id'), primary_key=True)
    resolved: Mapped[int] = mapped_column()

    history: Mapped['ChatHistory'] = relationship(back_populates='resolved')

    def __repr__(self):
        return f'ChatResolved(id={self.id!r}, resolved={self.resolved!r})'


class ChatFeedback(db.Model):
    __tablename__ = 'chat_feedback'

    id: Mapped[int] = mapped_column(ForeignKey('chat_history.id'), primary_key=True)
    stars: Mapped[int] = mapped_column()

    history: Mapped['ChatHistory'] = relationship(back_populates='feedback')

    def __repr__(self):
        return f'ChatFeedback(id={self.id!r}, stars={self.stars!r})'


class ChatHistory(db.Model):
    __tablename__ = 'chat_history'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    statuses: Mapped[List['ChatStatus']] = relationship(
        back_populates='history', cascade='all, delete-orphan'
    )

    messages: Mapped[List['ChatMessage']] = relationship(
        back_populates='history', cascade='all, delete-orphan', order_by=ChatMessage.time_sent
    )

    resolved: Mapped['ChatResolved'] = relationship(back_populates='history')
    feedback: Mapped['ChatFeedback'] = relationship(back_populates='history')

    def __repr__(self):
        return f'ChatHistory(id={self.id!r})'
