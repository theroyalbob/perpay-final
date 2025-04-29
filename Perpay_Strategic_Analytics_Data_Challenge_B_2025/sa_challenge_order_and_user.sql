--user data
select
	b.uuid as user_id,
	a.created_dt as signup_dt,
	a.company_name,
	a.spending_limit_est,
	a.valid_phone_ind,
	a.last_login,
	a.was_referred_ind,
	a.first_paystub_dt,
	a.first_application_start_ts,
	a.first_application_complete_ts,
	a.first_awaiting_payment_ts,
	a.first_repayment_ts
from domo_ext.user_data_model a
left join public.user b
	on a.user_id = b.id
where cast(a.created_dt as date) between '2020-10-13' and '2020-11-06';

--order data
select
	c.uuid as order_id,
	b.uuid as user_id,
	a.amount,
	a.number_of_payments,
	a.user_pinwheel_eligible_at_ap,
	a.approval_type,
	a.application_start_ts,
	a.application_complete_ts,
	a.awaiting_payment_ts,
	a.repayment_ts,
	a.canceled_ts,
	a.cancellation_type,
	a.risk_tier_at_uw
from domo_ext.order_data_model a
left join public.user b
	on a.user_id = b.id
left join public.order c
	on a.order_id = c.id
where risk_tier_at_uw = 'T0'
and application_start_ts::date between '2020-10-13' and '2020-11-06';




